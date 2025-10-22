import numpy as np
import pandas as pd
import yaml
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def load_data(filepath : str) -> pd.DataFrame:
    try:
         return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")


def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
    

def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

        cm = confusion_matrix(y_test, y_pred)
        cm.tolist()
        
        metrics_dict = {
            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict, cm
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")

def save_confusion_matrix(cm, path: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = "data/processed/test_processed.csv"
        train_data_path = "data/processed/train_processed.csv"

        model_path = "models/model.pkl"
        metrics_path = "metrics.json"

        test_data = load_data(test_data_path)
        train_data = load_data(train_data_path)
        X_test, y_test = prepare_data(test_data)

        test_df = mlflow.data.from_pandas(test_data)
        train_df = mlflow.data.from_pandas(train_data)

        model = load_model(model_path)
        metrics, cm = evaluation_model(model, X_test,y_test)
        save_metrics(metrics, metrics_path)
        cm_path = "confusion_matrix.png"
        save_confusion_matrix(cm, cm_path)

        n_estimators = load_params("params.yaml")
    except Exception as e:
        raise Exception(f"An Error occurred: {e}")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # здесь подумать над тем как менять имя на имя модели
    mlflow.set_experiment("water_quality_RandomForest")


    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_input(train_df, "train")
        mlflow.log_input(test_df, "test")
        mlflow.log_params({"n_estimators":n_estimators})
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import yaml
import mlflow, mlflow.sklearn

import pickle
import json

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

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

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")


def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = "data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "metrics.json"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model, X_test,y_test)
        save_metrics(metrics, metrics_path)

        n_estimators = load_params("params.yaml")
    except Exception as e:
        raise Exception(f"An Error occurred: {e}")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # здесь подумать над тем как менять имя на имя модели
    mlflow.set_experiment("water_quality_RandomForest")

    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_params({"n_estimators":n_estimators})
        mlflow.log_artifact(metrics_path)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
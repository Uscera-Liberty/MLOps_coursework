from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel


class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

app = FastAPI(
    title= "Example of Project",
    description = "predicting"
    )

with open ("/home/kovoya/web-back/DVC/models/model.pkl","rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to example predict"

@app.post("/predict")
def model_predict(water: Water):
    cols = [
        'ph','Hardness','Solids','Chloramines','Sulfate',
        'Conductivity','Organic_carbon','Trihalomethanes','Turbidity'
    ]
    sample = pd.DataFrame([{c:getattr(water,c) for c in cols}])

    X = sample[cols].values
    preds = model.predict(X)
    return {"prediction": preds.tolist()}

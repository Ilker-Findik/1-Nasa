from fastapi import FastAPI,Response,Request
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

with open("modelnasa.pkl","rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    scaler = saved_data["scaler"]
class AirqualityFeatures(BaseModel):
    City: object
    Temperature: float
    Humidity: float
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    Proximity_to_Industrial_Areas: float
    Population_Density: int


@app.post("/predict")
async def predict(features: AirqualityFeatures):
    model_input = features.model_dump(exclude={'City'})
    input_data = pd.DataFrame([model_input])
    print(input_data)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return {"prediction": prediction[0]}

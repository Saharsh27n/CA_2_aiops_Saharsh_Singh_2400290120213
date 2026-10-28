from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import features  # noqa: F401  (needed so joblib can unpickle build_features)

MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Delivery Time Prediction API")


class Order(BaseModel):
    Delivery_person_ID: Optional[str] = None
    Delivery_person_Age: Optional[float] = Field(None, examples=[34])
    Delivery_person_Ratings: Optional[float] = Field(None, examples=[4.5])
    Restaurant_latitude: float = Field(..., examples=[12.913041])
    Restaurant_longitude: float = Field(..., examples=[77.683237])
    Delivery_location_latitude: float = Field(..., examples=[13.043041])
    Delivery_location_longitude: float = Field(..., examples=[77.813237])
    Order_Date: str = Field(..., description="dd-mm-yyyy", examples=["25-03-2022"])
    Time_Orderd: Optional[str] = Field(None, description="HH:MM:SS", examples=["19:45:00"])
    Time_Order_picked: Optional[str] = Field(None, description="HH:MM:SS", examples=["19:50:00"])
    Weatherconditions: Optional[str] = Field(None, examples=["conditions Stormy"])
    Road_traffic_density: Optional[str] = Field(None, examples=["Jam"])
    Vehicle_condition: int = Field(..., examples=[2])
    Type_of_order: str = Field(..., examples=["Snack"])
    Type_of_vehicle: str = Field(..., examples=["scooter"])
    multiple_deliveries: Optional[float] = Field(None, examples=[1])
    Festival: Optional[str] = Field(None, examples=["No"])
    City: Optional[str] = Field(None, examples=["Metropolitian"])


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: Order):
    df = pd.DataFrame([order.model_dump()])
    minutes = float(model.predict(df)[0])
    return {"predicted_time_taken_min": round(minutes, 2)}

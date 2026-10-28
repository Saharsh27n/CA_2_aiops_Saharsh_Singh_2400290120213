"""Shared cleaning / feature engineering, used by the notebook and the API."""
import numpy as np
import pandas as pd

NUM_COLS = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "distance_km", "order_hour", "order_dayofweek",
    "pickup_wait_min",
]
CAT_COLS = [
    "Weatherconditions", "Road_traffic_density", "Type_of_order",
    "Type_of_vehicle", "Festival", "City",
]


def parse_target(series: pd.Series) -> pd.Series:
    """'(min) 24' -> 24.0"""
    return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip whitespace and turn 'NaN' strings into real missing values
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("object").str.strip().replace("NaN", np.nan)

    df["Weatherconditions"] = (
        df["Weatherconditions"].str.replace("conditions ", "", regex=False).replace("NaN", np.nan)
    )
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
              "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"].abs(), df["Restaurant_longitude"].abs(),
        df["Delivery_location_latitude"].abs(), df["Delivery_location_longitude"].abs(),
    )

    date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    ordered = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    picked = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")
    df["order_hour"] = ordered.dt.hour.fillna(picked.dt.hour)
    df["order_dayofweek"] = date.dt.day_of_week
    wait = (picked - ordered).dt.total_seconds() / 60
    df["pickup_wait_min"] = wait.where(wait >= 0, wait + 24 * 60)  # crossed midnight

    return df[NUM_COLS + CAT_COLS]

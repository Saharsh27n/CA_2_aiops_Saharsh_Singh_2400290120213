import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_ORDER = {
    "Delivery_person_Age": 34,
    "Delivery_person_Ratings": 4.5,
    "Restaurant_latitude": 12.913041,
    "Restaurant_longitude": 77.683237,
    "Delivery_location_latitude": 13.043041,
    "Delivery_location_longitude": 77.813237,
    "Order_Date": "25-03-2022",
    "Time_Orderd": "19:45:00",
    "Time_Order_picked": "19:50:00",
    "Weatherconditions": "conditions Stormy",
    "Road_traffic_density": "Jam",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "scooter",
    "multiple_deliveries": 1,
    "Festival": "No",
    "City": "Metropolitian",
}

MINIMAL_ORDER = {
    "Restaurant_latitude": 12.9,
    "Restaurant_longitude": 77.6,
    "Delivery_location_latitude": 13.0,
    "Delivery_location_longitude": 77.7,
    "Order_Date": "25-03-2022",
    "Vehicle_condition": 1,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
}


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_full_order():
    r = client.post("/predict", json=VALID_ORDER)
    assert r.status_code == 200
    minutes = r.json()["predicted_time_taken_min"]
    assert isinstance(minutes, float)
    assert 10 <= minutes <= 60  # training targets range 10-54 min


def test_predict_minimal_order_uses_imputation():
    r = client.post("/predict", json=MINIMAL_ORDER)
    assert r.status_code == 200
    assert r.json()["predicted_time_taken_min"] > 0


def test_predict_is_deterministic():
    a = client.post("/predict", json=VALID_ORDER).json()
    b = client.post("/predict", json=VALID_ORDER).json()
    assert a == b


@pytest.mark.parametrize("missing", [
    "Restaurant_latitude", "Delivery_location_longitude", "Order_Date",
    "Vehicle_condition", "Type_of_order", "Type_of_vehicle",
])
def test_missing_required_field_returns_422(missing):
    body = {k: v for k, v in MINIMAL_ORDER.items() if k != missing}
    assert client.post("/predict", json=body).status_code == 422


def test_wrong_type_returns_422():
    body = dict(MINIMAL_ORDER, Restaurant_latitude="not-a-number")
    assert client.post("/predict", json=body).status_code == 422


def test_empty_body_returns_422():
    assert client.post("/predict", json={}).status_code == 422

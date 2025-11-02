from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="IRIS Model API")

# Load trained model
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "Iris Prediction API is running!"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]).reshape(1, -1)

    prediction = model.predict(data)[0]
    return {"prediction": prediction}

import os
import joblib
from fastapi.testclient import TestClient
from google.cloud import storage
from app import app

MODEL_PATH = "model.joblib"
GCS_MODEL_PATH = "gs://mlops-week1/week6/model.joblib"

# Helper function to download model from GCS
def download_model_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"✅ Downloaded {source_blob_name} from {bucket_name} to {destination_file_name}")

def setup_module(module):
    """Download model from GCS before running tests"""
    if not os.path.exists(MODEL_PATH):
        # Parse bucket and blob names from GCS path
        gcs_parts = GCS_MODEL_PATH.replace("gs://", "").split("/", 1)
        bucket_name, blob_name = gcs_parts[0], gcs_parts[1]

        download_model_from_gcs(bucket_name, blob_name, MODEL_PATH)

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Iris Prediction API is running!"}

def test_predict():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

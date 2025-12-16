from feast import FeatureStore
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn

# --------------------------
# MLflow setup
# --------------------------
mlflow.set_tracking_uri("http://136.112.219.192:8100")

model_name = "iris_rf_model"
experiment_name = "mlops_iris_random_forest1"
client = MlflowClient()

# Get all registered versions of this model
versions = client.search_model_versions(f"name='{model_name}'")

# Pick latest version number
latest_version = max(int(v.version) for v in versions)
print(f"Loading latest model version: {latest_version}")

model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")

print("Model loaded successfully from MLflow.")

# --------------------------
# Fetch features from Feast
# --------------------------
store = FeatureStore(repo_path="/home/jupyter/week3/Assignment/iris_feature_repo/feature_repo")

entity_rows = [{"iris_id": 1001}, {"iris_id": 1002}]
features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species",
    ],
    entity_rows=entity_rows,
).to_df()

print("✅ Fetched features for inference:")
print(features)

# --------------------------
# Predict
# --------------------------
X = features[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
preds = model.predict(X)

results = pd.DataFrame({"iris_id": features["iris_id"], "predicted_species": preds})
print("\n🧾 Inference results:")
print(results)

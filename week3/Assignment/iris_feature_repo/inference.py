from feast import FeatureStore
import pandas as pd
import joblib

store = FeatureStore(repo_path="/home/jupyter/week3/Assignment/iris_feature_repo/feature_repo")

# Example new flower to predict
request_df = pd.DataFrame([
    [{"iris_id": 1001}, {"iris_id": 1002}, {"iris_id": 1003}]
])

# Fetch latest online features
features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species",
    ],
    entity_rows=[{"iris_id": 1001}, {"iris_id": 1002}, {"iris_id": 1003}],
).to_df()

print("Fetched features for inference:")
print(features)

model = joblib.load("/home/jupyter/week3/Assignment/iris_feature_repo/output_artifacts/model.joblib")
print("Model loaded successfully.")

X = features[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
preds = model.predict(X)

results = pd.DataFrame({"iris_id": features["iris_id"], "predicted_species": preds})
print("\nInference results:")
print(results)

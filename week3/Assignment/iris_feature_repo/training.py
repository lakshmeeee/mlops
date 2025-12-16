from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize store
store = FeatureStore(repo_path="/home/jupyter/week3/Assignment/iris_feature_repo/feature_repo")

# Entity DataFrame (usually from same BigQuery table)
entity_df = pd.read_gbq("SELECT iris_id, event_timestamp FROM `sheetgpt-385916.iris_dataset.iris_table`")

# Fetch features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species",
    ],
).to_df()

print("Training data from Feast + BigQuery:")
print(training_df.head())

X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_val, y_val)
print(f"Validation accuracy: {acc:.4f}")

model_path = "/home/jupyter/week3/Assignment/iris_feature_repo/output_artifacts/model.joblib"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

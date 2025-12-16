from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# --------------------------
# MLflow setup
# --------------------------
mlflow.set_tracking_uri("http://136.112.219.192:8100")
mlflow.set_experiment("mlops_iris_random_forest1")

# --------------------------
# Load features from Feast
# --------------------------
store = FeatureStore(repo_path="/home/jupyter/week3/Assignment/iris_feature_repo/feature_repo")

entity_df = pd.read_gbq("SELECT iris_id, event_timestamp FROM `sheetgpt-385916.iris_dataset.iris_table`")

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

print("Training data fetched from Feast + BigQuery:")
print(training_df.head())

# --------------------------
# Prepare data
# --------------------------
X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# --------------------------
# Hyperparameter tuning
# --------------------------
n_estimators_list = [50, 100, 150]
max_depth_list = [3, 5, 8]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():
            # Log params
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            print(f"\nModel (n={n_estimators}, depth={max_depth}) - acc={acc:.4f}, f1={f1:.4f}")
            
            signature = infer_signature(X_train, model.predict(X_train))

            # Log model to MLflow registry
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris_rf_model",
                signature=signature,
                input_example=X_train
            )

print("All experiments logged to MLflow.")

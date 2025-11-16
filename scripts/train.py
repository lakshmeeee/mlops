# train_with_mlflow_knn.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def load_data(path):
    df = pd.read_csv(path)
    X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
    y = df['species'].values
    return X, y


def run_training(data_path, poison_fraction_label, experiment_name,
                 seed=42, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    X, y = load_data(data_path)

    # Split cleanly
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )

    # ------------------------------------------------------
    # FIX: Set tracking URI BEFORE set_experiment
    # ------------------------------------------------------
    mlflow.set_tracking_uri("http://136.111.119.163:8100")
    mlflow.set_experiment(experiment_name)
    # ------------------------------------------------------

    with mlflow.start_run(run_name=f"{poison_fraction_label}_{model_name}"):

        # Log parameters
        mlflow.log_param("poison_fraction", poison_fraction_label)
        mlflow.log_param("model_type", "KNN")
        mlflow.log_param("n_neighbors", 5)
        mlflow.log_param("seed", seed)

        # ------------------------------------------------------
        # MODEL: KNN (highly sensitive to label poisoning)
        # ------------------------------------------------------
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save model locally + to MLflow
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(model, model_path)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="knn_model",                  # MLflow now prefers "name"
            registered_model_name="iris_model_for_poisoning",
            signature=signature,
            input_example=X_train[:5]          # first 5 rows
        )

        mlflow.log_artifact(model_path)

        # Save metrics as CSV (MLflow artifact)
        metrics_df = pd.DataFrame([{
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "poison_fraction": poison_fraction_label
        }])

        metrics_file = os.path.join(
            output_dir,
            f"metrics_knn_poison_{str(poison_fraction_label).replace('.','_')}.csv"
        )

        metrics_df.to_csv(metrics_file, index=False)
        mlflow.log_artifact(metrics_file)

        print(f"Finished run. Model saved to {model_path} and metrics to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--poison-label", type=str, default="clean")
    parser.add_argument("--experiment", type=str, default="iris-poisoning-experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    run_training(
        args.data,
        args.poison_label,
        args.experiment,
        seed=args.seed,
        output_dir=args.output_dir
    )

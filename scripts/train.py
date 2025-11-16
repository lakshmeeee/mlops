# train_with_mlflow.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
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

def run_training(data_path, poison_fraction_label, experiment_name, epochs=30, batch_size=None, seed=42, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    X, y = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    classes = np.unique(y)

    # Use SGDClassifier with log loss -> Logistic Regression with epochs control
    model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=seed)  # we will loop epochs manually

    mlflow.set_tracking_uri("http://136.111.119.163:8100")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log run-level params
        mlflow.log_param("poison_fraction", poison_fraction_label)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("model_type", "SGDClassifier-log (Logistic Regression via SGD)")
        mlflow.log_param("seed", seed)

        epoch_rows = []
        # For reproducibility we can shuffle manually per epoch
        rng = np.random.RandomState(seed)
        n_samples = X_train.shape[0]
        if batch_size is None:
            batch_size = n_samples  # full-batch gradient -> equivalent to scikit logistic per epoch

        for ep in range(1, epochs+1):
            # shuffle
            idx = rng.permutation(n_samples)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            # iterate mini-batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                # partial_fit needed for first call with classes
                if ep == 1 and start == 0:
                    model.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    model.partial_fit(X_batch, y_batch)

            # evaluate after epoch
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            train_acc = metrics.accuracy_score(y_train, y_pred_train)
            val_acc = metrics.accuracy_score(y_val, y_pred_val)
            # log per-epoch metrics
            mlflow.log_metric("train_accuracy", train_acc, step=ep)
            mlflow.log_metric("val_accuracy", val_acc, step=ep)

            epoch_rows.append({"epoch": ep, "train_accuracy": train_acc, "val_accuracy": val_acc})

            print(f"Epoch {ep}/{epochs} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

        # Save model locally & log artifact
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(model, model_path)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris_model_for_poisoning",
                signature=signature,
                input_example=X_train
            )
        mlflow.log_artifact(model_path)

        # Save epoch metrics CSV and log as artifact
        metrics_df = pd.DataFrame(epoch_rows)
        metrics_file = os.path.join(output_dir, f"metrics_poison_{str(poison_fraction_label).replace('.','_')}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        mlflow.log_artifact(metrics_file)

        print(f"Finished run. Model saved to {model_path} and metrics to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV data path (poisoned or clean)")
    parser.add_argument("--poison-label", type=str, default="clean", help="Label to describe dataset (e.g. 5pct)")
    parser.add_argument("--experiment", type=str, default="iris-poisoning-experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    run_training(args.data, args.poison_label, args.experiment, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, output_dir=args.output_dir)

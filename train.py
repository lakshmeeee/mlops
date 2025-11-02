from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from sklearn import metrics
from google.cloud import storage
import os

if __name__ == '__main__':

    DATASET_LOCAL_PATH = "./data/data.csv"
    LOCAL_MODEL_FILEPATH = "model.joblib"
    GCS_BUCKET_NAME = "mlops-week1"
    GCS_MODEL_PATH = "week6/model.joblib"

    # Read dataset1
    data = pd.read_csv(DATASET_LOCAL_PATH)
    
    # Train/test split
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    x_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train.species
    x_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    # Save model locally
    joblib.dump(model, LOCAL_MODEL_FILEPATH)

    # Evaluate and log accuracy
    prediction = model.predict(x_test)
    log_message = f'The accuracy of the Model is {metrics.accuracy_score(prediction, y_test):.3f}'
    with open('metrics.log', 'w') as f:
        f.write(log_message)

    # Upload model to GCS
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_MODEL_PATH)
    blob.upload_from_filename(LOCAL_MODEL_FILEPATH)

    print(f"Model uploaded to gs://{GCS_BUCKET_NAME}/{GCS_MODEL_PATH}")

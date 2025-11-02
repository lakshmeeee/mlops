from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from sklearn import metrics
    
if __name__ == '__main__':

    DATASET_LOCAL_PATH = "./data/data.csv"
    LOCAL_MODEL_FILEPATH = "model.joblib"
    
    data = pd.read_csv(DATASET_LOCAL_PATH)
    
    train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
    x_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train.species
    x_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    joblib.dump(model, LOCAL_MODEL_FILEPATH)

    prediction=model.predict(x_test)

    log_message = f'The accuracy of the Model is {metrics.accuracy_score(prediction, y_test):.3f}'

    with open('metrics.log', 'w') as f:
        f.write(log_message)

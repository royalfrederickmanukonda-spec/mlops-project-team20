import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR, MLFLOW_URI
from model_utils import save_model
import joblib
import argparse

mlflow.set_tracking_uri(MLFLOW_URI)
print("Checking TRAIN:")
train = pd.read_csv("data/processed/train.csv")
print(train.shape)
print(train.head())

def train(train_path, test_path, n_estimators=100, max_depth=None):
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Example: last column is target
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    with mlflow.start_run():
        mlflow.set_tag("mlflow.user", "Royal Frederick")
        mlflow.set_tag("mlflow.runName", "MLOPS Project Team 20")

        
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        mlflow.log_metric("accuracy", float(acc))

        mlflow.sklearn.log_model(model, "model")
        print("Accuracy:", acc)
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(PROCESSED_DIR/"train.csv"))
    parser.add_argument("--test", default=str(PROCESSED_DIR/"test.csv"))
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()
    train(args.train, args.test, args.n_estimators, args.max_depth)

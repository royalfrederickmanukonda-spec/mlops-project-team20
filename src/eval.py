import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import MODEL_DIR, PROCESSED_DIR, MLFLOW_URI
from model_utils import load_model
import argparse
import json

mlflow.set_tracking_uri(MLFLOW_URI)

def evaluate(model_path, test_path):
    # Load model
    model = load_model(model_path)

    # Load data
    test_df = pd.read_csv(test_path)

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Predict
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1_score", f1)

        # Save confusion matrix
        cm_path = MODEL_DIR / "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump(cm.tolist(), f)

        mlflow.log_artifact(str(cm_path))

    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=str(MODEL_DIR / "model.joblib"))
    parser.add_argument("--test", default=str(PROCESSED_DIR / "test.csv"))

    args = parser.parse_args()

    evaluate(args.model, args.test)

import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, name="model.joblib"):
    path = MODEL_DIR / name
    joblib.dump(model, path)
    return path

def load_model(path):
    return joblib.load(path)

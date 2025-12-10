from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from config import MODEL_DIR

app = FastAPI()

MODEL_PATH = MODEL_DIR / "model.joblib"

class InputData(BaseModel):
    data: list  # list of feature lists or dicts depending on implementation

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(payload: InputData):
    df = pd.DataFrame(payload.data)
    preds = model.predict(df).tolist()
    return {"predictions": preds}

@app.get("/health")
def health():
    return {"status":"ok"}

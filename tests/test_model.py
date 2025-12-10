import os
import pandas as pd
from data_prep import preprocess
from train import train
from predict import predict
from model_utils import save_model, load_model
from config import PROCESSED_DIR, MODEL_DIR

# ---------- TEST 1: Data Preprocessing ----------
def test_preprocess_removes_nans():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [5, 6, 7]
    })

    out = preprocess(df)
    assert out.isna().sum().sum() == 0, "Preprocess should remove NaN

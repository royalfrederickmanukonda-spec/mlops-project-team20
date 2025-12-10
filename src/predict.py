import pandas as pd
import joblib
from config import MODEL_DIR
from model_utils import load_model

def predict(model_path, input_data):
    """
    input_data can be:
    - A dictionary {"col1": value1, "col2": value2, ...}
    - A list of dicts
    - A list of lists
    """

    # Load model
    model = load_model(model_path

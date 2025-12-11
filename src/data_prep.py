import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DIR = Path("C:/Users/Royal/mlops-project-team20/data/raw")
PROCESSED_DIR = Path("C:/Users/Royal/mlops-project-team20/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_raw(path: Path):
    df = pd.read_csv(path, sep=';')
    print("Raw shape:", df.shape)
    return df

def preprocess(df):
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Convert categorical features to numeric
    df = pd.get_dummies(df, drop_first=True)

    print("After encoding:", df.shape)
    return df

def save_splits(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    print("Train:", train.shape)
    print("Test:", test.shape)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

if __name__ == "__main__":
    df = load_raw(RAW_DIR / "dataset.csv")
    df = preprocess(df)
    save_splits(df)

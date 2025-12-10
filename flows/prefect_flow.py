from prefect import flow, task
import subprocess

@task
def preprocess():
    subprocess.run(["python", "src/data_prep.py"], check=True)

@task
def train():
    subprocess.run(["python", "src/train.py", "--train", "data/processed/train.csv", "--test", "data/processed/test.csv"], check=True)

@flow
def ml_pipeline():
    preprocess()
    train()

if __name__ == "__main__":
    ml_pipeline()

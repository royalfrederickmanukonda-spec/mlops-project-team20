from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from config import PROCESSED_DIR

def make_report(reference_path, production_path, out_html="evidently_report.html"):
    ref = pd.read_csv(reference_path)
    prod = pd.read_csv(production_path)
    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref, prod)
    dashboard.save(out_html)
    print("Saved Evidently report to", out_html)

if __name__ == "__main__":
    make_report("data/processed/train.csv", "data/processed/test.csv")

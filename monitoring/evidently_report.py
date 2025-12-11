# monitoring/evidently_report.py
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def make_report(reference_path, production_path, out_html="evidently_report.html"):
    reference_df = pd.read_csv(reference_path)
    production_df = pd.read_csv(production_path)

    report = Report(metrics=[DataDriftPreset(drift_share=0.1)])


    # run the report and capture the returned run result (snapshot)
    run_result = report.run(reference_data=reference_df, current_data=production_df)

    # 1) preferred: run_result.save_html(path)
    if hasattr(run_result, "save_html"):
        run_result.save_html(out_html)
        print("Saved HTML report via run_result.save_html ->", out_html)
        return

    # 2) alternative: run_result.as_html() returns HTML string
    if hasattr(run_result, "as_html"):
        html = run_result.as_html()
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        print("Saved HTML report via run_result.as_html() ->", out_html)
        return

    # 3) fallback: get Python dict / JSON and write it
    if hasattr(run_result, "as_dict"):
        rep = run_result.as_dict()
        json_path = out_html.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print("save_html/as_html not available — wrote JSON snapshot ->", json_path)
        return

    # 4) last resort: try to serialize repr
    json_path = out_html.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"report_repr": repr(run_result)}, f)
    print("No known export methods found on run result — wrote repr ->", json_path)


if __name__ == "__main__":
    make_report(
        "C:/Users/Royal/mlops-project-team20/data/processed/train.csv",
        "C:/Users/Royal/mlops-project-team20/data/processed/test.csv",
        out_html="evidently_report.html",
    )

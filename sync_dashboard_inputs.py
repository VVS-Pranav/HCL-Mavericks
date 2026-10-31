"""Refresh dashboard inputs after running the three notebooks."""
from pathlib import Path
import shutil
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT if (ROOT / "prepared_data/model_data.joblib").exists() else ROOT.parent
source = PROJECT / "prepared_data/model_data.joblib"
if not source.exists():
    raise FileNotFoundError("Place this script beside the notebooks or in a folder immediately below them. Run Step 1 first.")

data = joblib.load(source)
output = ROOT / "dashboard_inputs"
output.mkdir(exist_ok=True)

latest = data["latest_metadata"][["customer_id", "forecast_month"]].reset_index(drop=True)
features = data["X_latest"].reset_index(drop=True)
pd.concat([latest, features], axis=1).to_csv(output / "latest_customer_features.csv", index=False)

history = pd.concat([
    data[f"{split}_metadata"][["customer_id", "billing_month", "bill_amount"]]
    for split in ["train", "validation", "test", "latest"]
], ignore_index=True)
history = history.drop_duplicates(["customer_id", "billing_month"]).sort_values(["customer_id", "billing_month"])
history.to_csv(output / "customer_history.csv", index=False)

if PROJECT != ROOT:
    for folder in ["model_outputs", "monitoring_outputs"]:
        destination = ROOT / folder
        destination.mkdir(exist_ok=True)
        for path in (PROJECT / folder).iterdir():
            if path.is_file() and path.suffix in [".csv", ".json", ".joblib"]:
                shutil.copy2(path, destination / path.name)
print(f"Updated dashboard inputs for {len(latest):,} customers.")

# src/pipelines/train_pipeline.py
from src.utils.seed import set_seed
from src.data.make_synthetic import make_dataset
from src.data.clean import clean
from src.data.check import run_checks
from src.data.split import split_train_val_test
from src.models.train_regressor import train_model
from src.models.evaluate import evaluate_model

import joblib
import os
import pandas as pd

def main():
    set_seed(42) 

    # File locations
    raw_path   = "data/raw/ev_battery_health.csv"
    clean_path = "data/interim/clean.csv"
    out_dir    = "data/processed"

    print("1) Generating synthetic data …")
    make_dataset(out_path=raw_path)

    print("2) Cleaning …")
    clean(in_path=raw_path, out_path=clean_path)

    print("3) Data checks …")
    run_checks(in_path=clean_path)

    print("4) Splitting (train/val/test) …")
    paths = split_train_val_test(in_path=clean_path, out_dir=out_dir)

    print("5) Training (RandomForest) …")
    model, run_id = train_model(paths, params_path="configs/params.yaml")
    print(f"   -> MLflow run_id: {run_id}")

    # Save with joblib
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/random_forest.joblib")

    train_df = pd.read_csv(paths["train"])
    feature_names = train_df.drop(columns=["state_of_health"]).columns.tolist()
    with open("artifacts/feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    print("6) Evaluating on TEST …")
    evaluate_model(model, paths, run_id=run_id)

    print("Done.")

if __name__ == "__main__":
    main()

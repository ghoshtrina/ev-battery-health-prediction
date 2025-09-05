import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "rmse": rmse,
        "mae" : mean_absolute_error(y_true, y_pred),
        "r2"  : r2_score(y_true, y_pred),
    }

def train_model(paths: dict, params_path: str = "configs/params.yaml"):
    """
    Trains a RandomForestRegressor on train.csv, evaluates on val.csv,
    logs everything to MLflow, and returns (model, mlflow_run_id).
    """
    # --- load data ---
    train_df = pd.read_csv(paths["train"])
    val_df   = pd.read_csv(paths["val"])

    X_train = train_df.drop(columns=["state_of_health"])
    y_train = train_df["state_of_health"]
    X_val   = val_df.drop(columns=["state_of_health"])
    y_val   = val_df["state_of_health"]

    # --- load params (with forgiving defaults) ---
    params       = yaml.safe_load(open(params_path))
    model_cfg    = params.get("model", {})
    n_estimators = model_cfg.get("n_estimators", 300)
    max_depth    = model_cfg.get("max_depth", None)
    n_jobs       = model_cfg.get("n_jobs", -1)
    seed         = params.get("random_seed", 42)

    # --- model ---
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=seed,
    )

    # --- MLflow logging ---
    mlflow.set_experiment("ev-battery-soh")
    with mlflow.start_run() as run:
        # fit
        rf.fit(X_train, y_train)

        # predictions
        y_pred_tr = rf.predict(X_train)
        y_pred_va = rf.predict(X_val)

        # metrics
        tr = regression_metrics(y_train, y_pred_tr)
        va = regression_metrics(y_val,   y_pred_va)

        # log params + metrics
        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_jobs": n_jobs,
            "random_seed": seed,
        })
        mlflow.log_metrics({
            "train_rmse": tr["rmse"], "train_mae": tr["mae"], "train_r2": tr["r2"],
            "val_rmse":   va["rmse"], "val_mae":   va["mae"], "val_r2":   va["r2"],
        })

        # feature names (handy)
        mlflow.log_text("\n".join(X_train.columns.tolist()), "feature_names.txt")

        # save model
        mlflow.sklearn.log_model(rf, artifact_path="model")

        run_id = run.info.run_id

    # quick console print
    print(f"[Train] RMSE={tr['rmse']:.3f}  MAE={tr['mae']:.3f}  R2={tr['r2']:.3f}")
    print(f"[ Val ] RMSE={va['rmse']:.3f}  MAE={va['mae']:.3f}  R2={va['r2']:.3f}")

    return rf, run_id

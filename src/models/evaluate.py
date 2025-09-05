import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from src.models.postprocess import bucketize
from src.models.train_regressor import regression_metrics

def _start_run_compat(run_id: str):
    # Newer MLflow
    try:
        return mlflow.start_run(run_id=run_id)
    except TypeError:
        # Older MLflow expects run_uuid kwarg
        return mlflow.start_run(run_uuid=run_id)

def evaluate_model(model, paths: dict, run_id: str):
    """
    Evaluate on TEST split, log metrics + derived bucket report to MLflow,
    and print a summary.
    """
    test_df = pd.read_csv(paths["test"])
    X_test = test_df.drop(columns=["state_of_health"])
    y_test = test_df["state_of_health"]

    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)

    # Derived classification from predicted SOH
    y_true_b = bucketize(y_test.values)
    y_pred_b = bucketize(y_pred)
    report   = classification_report(y_true_b, y_pred_b, digits=3)

    # Log to MLflow
    mlflow.set_experiment("ev-battery-soh")
    with _start_run_compat(run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_text(report, "derived_bucket_report.txt")

    print("[ Test] " + "  ".join(f"{k.upper()}={v:.3f}" for k, v in metrics.items()))
    print("\nDerived bucket classification (from predictions):\n")
    print(report)

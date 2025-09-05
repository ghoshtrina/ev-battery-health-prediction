import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from src.models.postprocess import bucketize

def _maybe_make_strata(y_series, enable: bool):
    """
    Build stratification labels from state_of_health if enabled.
    Returns either an array of bucket labels or None.
    """
    if not enable:
        return None

    # Derive buckets from continuous SOH using our postprocess thresholds
    labels = bucketize(y_series.values)

    # Quick sanity: if any bucket has < 2 samples, stratify will likely fail.
    # We'll disable stratification in that case to keep it beginner-friendly.
    import numpy as np
    unique, counts = np.unique(labels, return_counts=True)
    if (counts < 2).any():
        print("[WARN] Not enough samples in at least one bucket; falling back to non-stratified split.")
        return None
    return labels

def split_train_val_test(
    in_path: str,
    out_dir: str,
    params_path: str = "configs/params.yaml",
):
    """
    Reads cleaned data, performs train/val/test split, and writes CSVs to out_dir.
    - Uses random_seed, test_size, val_size from params.yaml
    - Optional: stratify_by_bucket (derived from state_of_health)
    Returns dict with paths.
    """
    df = pd.read_csv(in_path)
    params = yaml.safe_load(open(params_path))
    
    split_cfg = params.get('split', {})
    test_size = split_cfg.get('test_size', 0.2)
    val_size = split_cfg.get('val_size', 0.1)
    seed = params.get('random_seed', 42)
    stratify_flag = split_cfg.get('stratify_by_bucket', False)

    # Create optional stratification labels from SOH
    y = df["state_of_health"]
    strat_labels = _maybe_make_strata(y, enable=stratify_flag)

    # First: train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strat_labels if strat_labels is not None else None,
    )

    # Second: val from train
    # If stratification enabled, recompute labels on the new train slice
    if strat_labels is not None:
        train_labels = bucketize(train_df["state_of_health"].values)
        # Guard again for tiny buckets
        import numpy as np
        uniq, cnts = np.unique(train_labels, return_counts=True)
        if (cnts < 2).any():
            print("[WARN] Not enough samples to stratify validation split; using non-stratified val split.")
            train_labels = None
    else:
        train_labels = None

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_labels if train_labels is not None else None,
    )

    # Write outputs
    train_p = f"{out_dir}/train.csv"
    val_p   = f"{out_dir}/val.csv"
    test_p  = f"{out_dir}/test.csv"
    train_df.to_csv(train_p, index=False)
    val_df.to_csv(val_p, index=False)
    test_df.to_csv(test_p, index=False)

    # Small summary (handy when running the pipeline)
    def _counts(name, frame):
        print(f"{name:>5}: {len(frame):4d} rows", end="")
        # show bucket distribution for a quick eyeball
        buckets = bucketize(frame['state_of_health'].values)
        import numpy as np
        uniq, cnt = np.unique(buckets, return_counts=True)
        dist = ", ".join([f"{u}:{c}" for u, c in zip(uniq, cnt)])
        print(f"  | buckets → {dist}")

    print("\nSplit summary")
    _counts("train", train_df)
    _counts("  val", val_df)
    _counts(" test", test_df)
    print("")

    return {"train": train_p, "val": val_p, "test": test_p}

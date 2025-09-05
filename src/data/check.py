import pandas as pd
import yaml

def run_checks(
    in_path: str,
    schema_path: str = "configs/schema.yaml",
    strict: bool = True,
):
    """
    Validate a cleaned dataset against the schema:
      - required columns present
      - (optional) no unexpected columns if strict=True
      - dtypes: int vs float
      - value ranges: min/max (inclusive)
      - no missing values
      - duplicates: fail if strict=True, otherwise warn

    Returns True if all checks pass; raises AssertionError otherwise.
    """

    df = pd.read_csv(in_path)
    schema = yaml.safe_load(open(schema_path))['columns']

    # Column presence or extras
    expected = list(schema.keys())
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]

    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    if strict and extra:
        raise AssertionError(f"Extra columns: {extra}")
    
    for col, spec in schema.items():
        # All numeric values
        col_as_num = pd.to_numeric(df[col], errors="coerce") # coerce -> bad values become NaN
        if col_as_num.isna().any():
            bad_count = col_as_num.isna().sum()
            raise ValueError(
                f"Column '{col}' contains {bad_count} non-numeric value(s); "
                "ensure values are numeric."
            )
        
        # Enforce dtype (int vs float)
        if spec["type"] == "int":
            # If any value has a fractional part, this is an error.
            fractional_mask = (col_as_num != col_as_num.round())
            if fractional_mask.any():
                first_bad = col_as_num[fractional_mask].iloc[0]
                raise ValueError(
                    f"Column '{col}' must be integers only; found non-integer like {first_bad}."
                )
            df[col] = col_as_num.round().astype(int)
        else:  # float
            df[col] = col_as_num.astype(float)

        # Missing values
        na_count = int(df[col].isna().sum())
        if na_count > 0:
            raise ValueError(f"Column '{col}' has {na_count} missing value(s).")
        
        # Range checks (inclusive)
        mn, mx = spec["min"], spec["max"]
        out_of_range_mask = (df[col] < mn) | (df[col] > mx)
        if out_of_range_mask.any():
            bad_rows = int(out_of_range_mask.sum())
            example_val = df.loc[out_of_range_mask, col].iloc[0]
            raise ValueError(
                f"Column '{col}' has {bad_rows} value(s) outside [{mn}, {mx}]. "
                f"Example offending value: {example_val}"
            )
        
    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        if strict:
            raise ValueError(f"Found {dup_count} duplicate row(s) (strict=True).")
        else:
            print(f"[WARN] Found {dup_count} duplicate row(s); continuing (strict=False).")

    # If we get here, all checks passed
    return True

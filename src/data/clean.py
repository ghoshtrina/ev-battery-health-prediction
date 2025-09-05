import pandas as pd
import yaml

def clean(in_path: str, out_path: str, schema_path: str = "configs/schema.yaml"):
    df = pd.read_csv(in_path)
    schema = yaml.safe_load(open(schema_path))["columns"]

    for col, spec in schema.items():
        # enforce type
        if spec["type"] == "int":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)

        # clip values to schema min/max
        df[col] = df[col].clip(spec["min"], spec["max"])

    # remove duplicates
    df = df.drop_duplicates()

    df.to_csv(out_path, index=False)
    return out_path

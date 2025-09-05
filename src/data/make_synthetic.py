from src.utils.seed import set_seed
import numpy as np
import pandas as pd
import yaml

def make_dataset(
    out_path: str,
    n_samples: int = 5000,
    params_path: str = "configs/params.yaml",
    schema_path: str = "configs/schema.yaml"
):
    """
    Generate a synthetic EV battery dataset that respects the schema ranges.
    - Seed comes from params.yaml (falls back to 42 once, centrally).
    - Columns & ranges come from schema.yaml.
    """
    # --- load config ---
    params = yaml.safe_load(open(params_path))
    schema = yaml.safe_load(open(schema_path))["columns"]

     # --- seeding ---
    cfg_seed = params.get("random_seed")          # may be None
    seed = cfg_seed if cfg_seed is not None else 42
    set_seed(seed)                                # sets numpy + python.random
    rng = np.random.default_rng(seed)             # same fallback used here

    # --- feature generation using schema bounds ---
    cycle_count = rng.integers(
        schema["cycle_count"]["min"],
        schema["cycle_count"]["max"],
        n_samples,
    )

    avg_temperature = rng.normal(30, 7, n_samples)
    avg_temperature = np.clip(
    avg_temperature,
    schema["avg_temperature"]["min"],
    schema["avg_temperature"]["max"],
    )
   
    charge_rate = rng.normal(1.0, 0.5, n_samples) 
    charge_rate = np.clip(
        charge_rate,
        schema["charge_rate"]["min"],
        schema["charge_rate"]["max"],
    )

    discharge_rate = rng.normal(1.0, 0.5, n_samples) 
    discharge_rate = np.clip(
        discharge_rate,
        schema["discharge_rate"]["min"],
        schema["discharge_rate"]["max"],
    )

    depth_of_discharge = rng.uniform(
        schema["depth_of_discharge"]["min"],
        schema["depth_of_discharge"]["max"],
        n_samples,
    )

    internal_resistance = rng.normal(0.05, 0.01, n_samples) + cycle_count * 1e-5
    internal_resistance = np.clip(
        internal_resistance,
        schema["internal_resistance"]["min"],
        schema["internal_resistance"]["max"],
    )

    # --- target (SOH) ---
    soh = (
        100
        - 0.005 * cycle_count
        - 0.02 * (avg_temperature - 25)
        - 0.5 * (charge_rate - 1.0)
        - 0.3 * (discharge_rate - 1.0)
        - 0.02 * (depth_of_discharge - 50)
        - 50 * internal_resistance
    )

     # --- target (SOH) ---
    soh = (
        100
        - 0.005 * cycle_count
        - 0.02 * (avg_temperature - 25)
        - 0.5 * (charge_rate - 1.0)
        - 0.3 * (discharge_rate - 1.0)
        - 0.02 * (depth_of_discharge - 50)
        - 50 * internal_resistance
    )
    soh = soh + rng.normal(0, 2, n_samples)
    soh = np.clip(
        soh,
        schema["state_of_health"]["min"],
        schema["state_of_health"]["max"],
    )

    # --- build & save ---
    df = pd.DataFrame(
        {
            "cycle_count": cycle_count,
            "avg_temperature": avg_temperature,
            "charge_rate": charge_rate,
            "discharge_rate": discharge_rate,
            "depth_of_discharge": depth_of_discharge,
            "internal_resistance": internal_resistance,
            "state_of_health": soh,
        }
    )
    df.to_csv(out_path, index=False)
    return out_path

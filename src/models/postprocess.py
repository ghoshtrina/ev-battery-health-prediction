import numpy as np

def bucketize(soh_array):
    s = np.asarray(soh_array)
    # Industry thresholds: Healthy ≥85, Moderate 70–84, EOL <70
    return np.select([s < 70, s < 85], ["EOL", "Moderate"], default="Healthy")

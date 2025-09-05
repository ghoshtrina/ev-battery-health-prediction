import numpy as np, random
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

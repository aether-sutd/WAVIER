import numpy as np
from scipy.stats import skew, kurtosis

def safe_stats(data):
    if len(data) < 2:
        return (0, 0, 0, 0)
    return (
        np.mean(data),
        np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
        skew(data),
        kurtosis(data)
    )
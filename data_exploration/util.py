from typing import List, Tuple
import numpy as np


def slope(x1: float, y1: float, x2: float, y2: float) -> float:
    return (y2 - y1) / (x2 - x1)


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def reject_pair_outliers(pairs, slopes, m=2):
    pairs = np.array(pairs)
    mean = np.mean(slopes)
    std = np.std(slopes)
    i = np.array([abs(pair[2] - mean) < m * std for pair in pairs])
    return pairs[i]

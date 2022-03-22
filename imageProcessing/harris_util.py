from typing import List

import numpy as np
import typing


def get_gaussian_kernel(window_size: int, sigma: int):
    x, y = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    d = np.sqrt(x * x + y * y)
    mu = 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    result: np.ndarray = np.array([0] * len(g[0]))
    for (i, row) in enumerate(g):
        result = np.add(result, row)

    return result / sum(result)

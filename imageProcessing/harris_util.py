from typing import List

import numpy as np
import typing


def get_gaussian_kernel(window_size: int, sigma: int):
    x, y = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    d = np.sqrt(x * x + y * y)
    mu = 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    # print("2D Gaussian-like array:")
    # print(g)

    r: List[float] = [0] * len(g[0])
    for (i, row) in enumerate(g):
        r += row

    return r / sum(r)


#    smoothing_3tap = [0.27901, 0.44198, 0.27901]
for i in range(10, 30):
    print(s := i / 25)
    print(get_gaussian_kernel(window_size=3, sigma=s))

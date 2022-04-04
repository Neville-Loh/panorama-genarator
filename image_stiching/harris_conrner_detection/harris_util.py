from typing import List, Tuple, Optional

import numpy as np
from imageProcessing.convolve2D import computeSeparableConvolution2DOddNTapBorderZero
import imageProcessing.convolve2D as IPConv2D
"""
Utility class contain helper function for computing harris corner

@Author Neville Loh
"""


# Default image type is np array.
ImageArray = np.ndarray


def sobel(px_array: ImageArray) -> Tuple[ImageArray, ImageArray]:
    """Compute the gaussian 1D kernel given the sigma as a constants
    Parameters
    ----------
    px_array : ImageArray
        generate a window_size by window_size filter

    Returns
    -------
    Tuple[ImageArray, ImageArray]
        A tuple containing 2 result image for along x and y direction
    """
    image_height, image_width = np.shape(px_array)
    ix_kernel = ([-1, 0, 1], [1, 2, 1])
    iy_kernel = ([1, 2, 1], [-1, 0, 1])

    i_x = computeSeparableConvolution2DOddNTapBorderZero(px_array, image_width, image_height,
                                                         kernelAlongX=ix_kernel[0],
                                                         kernelAlongY=ix_kernel[1])
    i_y = computeSeparableConvolution2DOddNTapBorderZero(px_array, image_width, image_height,
                                                         kernelAlongX=iy_kernel[0],
                                                         kernelAlongY=iy_kernel[1])
    return np.array(i_x), np.array(i_y)


def compute_gaussian_averaging(pixel_array: ImageArray, windows_size: Optional[int] = 5) -> ImageArray:
    """Compute the gaussian 1D kernel given the sigma as a constants
    Parameters
    ----------
    pixel_array : ImageArray
        A 2 dimensional imageArray that contain
    windows_size : Optional[int]
        the default windows size used for gaussian averaging, if none are supplied, a default size of 5 will be used

    Returns
    -------
    ImageArray
        The result image after gaussian filter is applied
    """
    image_height, image_width = np.shape(pixel_array)
    kernel = get_gaussian_kernel(windows_size, sigma=1)
    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
        pixel_array.tolist(), image_width, image_height, kernel)

    return np.array(averaged)


def get_gaussian_kernel(window_size: int, sigma: float, offset: Optional[float] = 0.0) -> List[float]:
    """Compute the gaussian 1D kernel given the sigma as a constant

    Parameters
    ----------
    window_size : int
        generate a window_size by window_size filter
    sigma : float
        the steepness of the fall of the Gaussian
    offset : float Optional
        the offset of the gaussian windows

    Returns
    -------
    list[float]
        a list of kernel of size window_size
    """
    x, y = np.meshgrid(np.linspace(-1, 1, window_size), np.linspace(-1, 1, window_size))
    d = np.sqrt(x * x + y * y)
    mu = offset
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    result: np.ndarray = np.array([0] * len(g[0]))
    for (i, row) in enumerate(g):
        result = np.add(result, row)

    return result / sum(result)

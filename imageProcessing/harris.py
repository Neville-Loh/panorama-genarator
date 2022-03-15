from typing import List, Tuple, Optional, Type
import numpy as np
import imageProcessing.utilities as IPUtils
import imageProcessing.convolve2D as IPConv2D

from imageProcessing.convolve2D import computeSeparableConvolution2DOddNTapBorderZero

ImageArray = List[List[float]]


class Corner:
    def __init__(self, x: int, y: int, cornerness: float):
        self.x = x
        self.y = y
        self.cornerness = cornerness


def sobel(px_array: ImageArray, image_width: int, image_height: int) -> Tuple[ImageArray, ImageArray]:
    """
    Apply sobel filter using 2D convolution
    Returns: image i_x, i_y
    """
    ix_kernel = ([-1, 0, 1], [1, 2, 1])
    iy_kernel = ([1, 2, 1], [-1, 0, 1])

    i_x = computeSeparableConvolution2DOddNTapBorderZero(px_array, image_width, image_height,
                                                         kernelAlongX=ix_kernel[0],
                                                         kernelAlongY=ix_kernel[1])
    i_y = computeSeparableConvolution2DOddNTapBorderZero(px_array, image_width, image_height,
                                                         kernelAlongX=iy_kernel[0],
                                                         kernelAlongY=iy_kernel[1])
    return i_x, i_y


def get_square_and_mixed_derivatives(i_x: ImageArray, i_y: ImageArray) -> Tuple[ImageArray, ImageArray, ImageArray]:
    i_x = np.array(i_x)
    i_y = np.array(i_y)
    return np.square(i_x).tolist(), np.square(i_y).tolist(), np.multiply(i_x, i_y).tolist()


def compute_gaussian_averaging(pixel_array: ImageArray, image_width: int,
                               image_height: int, windows_size: Optional[int] = 5) -> ImageArray:
    # You can customize GaussianBlur coefficient by: http://dev.theomader.com/gaussian-kernel-calculator
    SAMPLE_KERNEL = [0.1784, 0.210431, 0.222338, 0.210431, 0.1784]
    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
        pixel_array, image_width, image_height, SAMPLE_KERNEL)

    return averaged


def get_cornerness(ix2: ImageArray, iy2: ImageArray, ixiy: ImageArray, alpha: float) -> ImageArray:
    ix2, iy2, ixiy = np.array(ix2), np.array(iy2), np.array(ixiy)
    trace: ImageArray = np.square(np.add(ix2,iy2)) * alpha
    result = np.multiply(ix2,iy2) - np.square(ixiy) - trace
    return result


from typing import List, Tuple, Optional, Type, Any
from matplotlib import pyplot as plt
import numpy as np
import heapq

import imageProcessing.smoothing as IPSmooth
import imageProcessing.utilities as IPUtils
import imageProcessing.convolve2D as IPConv2D

from imageProcessing.convolve2D import computeSeparableConvolution2DOddNTapBorderZero
from imageProcessing.harris_util import get_gaussian_kernel

ImageArray = np.ndarray


class Corner:
    def __init__(self, index: Tuple[int, int], cornerness: float):
        self.x, self.y = index
        self.cornerness = cornerness

    def __lt__(self, other):
        return self.cornerness > other.cornerness

    def __eq__(self, other):
        return self.cornerness == other.cornerness

    def __str__(self) -> str:
        return str((self.x, self.y, self.cornerness))

    def __repr__(self):
        return str(self)


def compute_harris_corner(img_original: List[List[int]],
                          n_corner: Optional[int] = 5,
                          alpha: Optional[float] = 0.04,
                          gaussian_window_size: Optional[int] = 3,
                          plot_image: Optional[bool] = False) \
        -> List[Corner]:
    """
    Compute the harris corner for the picture
    return the corner activated value in decreasing value
    optional parameters
        n_corner: Optional[int], default =5,
        alpha: Optional[float], default =0.04,
        gaussian_window_size: Optional[int], default =3,
        plot_image: Optional[bool], default =False)

    """

    # step 1
    np_original = np.array(img_original)
    height, width = np.shape(np_original)
    px_array_left = IPSmooth.computeGaussianAveraging3x3(img_original, width, height)

    # Step 2
    # Implement e.g. Sobel filter in x and y, (The gradient)  for X and Y derivatives
    ix, iy = sobel(np_original)

    # Step 3
    # compute the square derivatives and the product of the mixed derivatives, smooth them,
    # Play with different size of gaussian window (5x5, 7x7, 9x9)

    ix2, iy2, ixiy = t_left = get_square_and_mixed_derivatives(ix, iy)

    # gaussian blur
    ix2_blur_left, iy2_blur_left, ixiy_blur_left = [compute_gaussian_averaging(img, windows_size=gaussian_window_size) for img in t_left]


    # Choose a Harris constant between 0.04 and 0.06

    # 5 extract Harris corners as (x,y) tuples in a data structure, which is sorted according to the strength of the
    # Harris response function C, sorted list of tuples
    corner_img_array = get_image_cornerness(ix2_blur_left, iy2_blur_left, ixiy_blur_left, alpha)

    # 5.5 non-max suppression
    corner_img_array = bruteforce_non_max_suppression(corner_img_array, window_size=3)

    # 6 Prepare n=1000 strongest conner per image
    pq_n_best_corner = heapq.nsmallest(n_corner, get_all_corner(corner_img_array))

    pq_n_best_corner_coor = [(corner.y, corner.x) for corner in pq_n_best_corner]

    if plot_image:
        plt.figure(figsize=(20, 18))
        plt.gray()
        plt.imshow(img_original)
        plt.scatter(*zip(*pq_n_best_corner_coor), s=1, color='r')
        plt.axis('off')
        plt.show()
    return pq_n_best_corner


def sobel(px_array: ImageArray) -> Tuple[ImageArray, ImageArray]:
    """
    Apply sobel filter using 2D convolution
    Returns: image i_x, i_y
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


def get_square_and_mixed_derivatives(i_x: ImageArray, i_y: ImageArray) -> Tuple[ImageArray, ImageArray, ImageArray]:
    return np.square(i_x), np.square(i_y), np.multiply(i_x, i_y)


def compute_gaussian_averaging(pixel_array: ImageArray, windows_size: Optional[int] = 5) -> ImageArray:
    image_height, image_width = np.shape(pixel_array)
    # You can customize GaussianBlur coefficient by: http://dev.theomader.com/gaussian-kernel-calculator
    #SAMPLE_KERNEL = [0.1784, 0.210431, 0.222338, 0.210431, 0.1784]
    kernel = get_gaussian_kernel(windows_size, sigma=1)
    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
        pixel_array.tolist(), image_width, image_height, kernel)

    return np.array(averaged)


def get_image_cornerness(ix2: ImageArray, iy2: ImageArray, ixiy: ImageArray, alpha: float) -> ImageArray:
    result = np.multiply(ix2, iy2) - np.square(ixiy) - (np.square(np.add(ix2, iy2)) * alpha)
    return result


def get_all_corner(img: ImageArray) -> List[Type[Corner]]:
    pq = []
    for index, val in np.ndenumerate(img):
        heapq.heappush(pq, Corner(index, val))
    return pq


def bruteforce_non_max_suppression(input_img: ImageArray, window_size: Optional[int] = 3) -> ImageArray:
    height, width = np.shape(input_img)
    center_window_index = window_size ** 2 // 2
    input_img = input_img.flatten()

    # Create window
    window = []
    for i in range(window_size):
        window += [i * width + j for j in range(window_size)]
    window = np.array(window)

    row = 0
    while window[-1] < len(input_img):
        max_index = np.argmax(input_img[window])

        # suppress non max to 0 if nearest neighbour have a higher activation
        if max_index != center_window_index:
            input_img[window[center_window_index]] = 0

        # shift to next row
        if window[0] + window_size > width * row + width - 1:
            window += window_size
            row += 1
        else:
            window += 1

    return input_img.reshape(height, width)

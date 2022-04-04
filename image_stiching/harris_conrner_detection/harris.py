import numpy as np
import heapq
import imageProcessing.smoothing as IPSmooth
from typing import List, Tuple, Optional, Type
from matplotlib import pyplot as plt
from image_stiching.corner import Corner, get_all_corner
from image_stiching.harris_conrner_detection.harris_util import sobel, compute_gaussian_averaging
from image_stiching.performance_evaulation.util import measure_elapsed_time

"""
Harris corner detection
A generic implementation of Harris Corner detection Algorithm

@Author: Neville Loh
"""

# Default image type is np array.
ImageArray = np.ndarray


@measure_elapsed_time
def compute_harris_corner(img_original: List[List[int]],
                          n_corner: Optional[int] = 5,
                          alpha: Optional[float] = 0.04,
                          gaussian_window_size: Optional[int] = 5,
                          plot_image: Optional[bool] = False) \
        -> List[Type[Corner]]:
    """
    Compute the harris corner for the picture
    return the corner activated value in decreasing value
    optional parameters
        n_corner: Optional[int], default =5,
        alpha: Optional[float], default =0.04,
        gaussian_window_size: Optional[int], default =5,
        plot_image: Optional[bool], default =False)

    """

    # Apply Gaussian filter, blur and smoothing for the input image
    np_original = np.array(img_original)
    height, width = np.shape(np_original)
    px_array = IPSmooth.computeGaussianAveraging3x3(img_original, width, height)

    # Apply Sobel filters in x and y direction to compute the gradient, X and Y derivatives
    ix, iy = sobel(px_array)

    # Compute the square derivatives and the product of the mixed derivatives, smooth them,
    result_tuple = get_square_and_mixed_derivatives(ix, iy)

    # Apply Gaussian blur with input windows size, if no window size is given, a default size of 5 by 5 is used
    ix2_blur_left, iy2_blur_left, ixiy_blur_left \
        = [compute_gaussian_averaging(img, windows_size=gaussian_window_size) for img in result_tuple]

    # Compute the Harris response for each pixel
    corner_img_array = get_image_cornerness(ix2_blur_left, iy2_blur_left, ixiy_blur_left, alpha)

    # Apply local non-max suppression for each piexels
    corner_img_array = bruteforce_non_max_suppression(corner_img_array, window_size=3)

    # Prepare n=1000 strongest conner per image
    pq_n_best_corner = heapq.nsmallest(n_corner, get_all_corner(corner_img_array))

    # Plot the image if optional argument plot_image is true
    if plot_image:
        plt.figure(figsize=(20, 18))
        plt.gray()
        plt.imshow(img_original)
        plt.scatter(*zip(*[(corner.x, corner.y) for corner in pq_n_best_corner]), s=1, color='r')
        plt.axis('off')
        plt.show()

    # Return List of Corner as heap
    return pq_n_best_corner


def get_square_and_mixed_derivatives(i_x: ImageArray, i_y: ImageArray) -> Tuple[ImageArray, ImageArray, ImageArray]:
    """Compute the square and mixed derivatives of the image
    Parameters
    ----------
    i_x : ImageArray
        An ImageArray that contain derivatives of x direction
    i_y : ImageArray
        An ImageArray that contain derivatives of y direction

    Returns
    -------
    Tuple[ImageArray, ImageArray, ImageArray]
        A tuple containing 3 result, the square of x derivatives, the square of x derivatives, and the product of the
        x y derivatives.
    """
    return np.square(i_x), np.square(i_y), np.multiply(i_x, i_y)


def get_image_cornerness(ix2: ImageArray, iy2: ImageArray, ixiy: ImageArray, alpha: float) -> ImageArray:
    """Compute the gaussian 1D kernel given the sigma as a constants
    Parameters
    ----------
    ix2 : ImageArray
        An ImageArray that contain squares derivatives of x direction
    iy2 : ImageArray
        An ImageArray that contain squares derivatives of y direction
    ixiy : ImageArray
        An ImageArray that contain product of the derivative
    alpha : int


    Returns
    -------
    ImageArray
        An ImageArray, where each coordinate contains the Harris response of each pixel.
    """
    result = np.multiply(ix2, iy2) - np.square(ixiy) - (np.square(np.add(ix2, iy2)) * alpha)
    return result


def bruteforce_non_max_suppression(input_img: ImageArray, window_size: Optional[int] = 3) -> ImageArray:
    """Applied local non max suppression for the iamge
    A n by n pixel windows is iterated over the image while the center pixel is suppressed if
    it is not a local maximum compare to its nearest 8 neighbour.

    Parameters
    ----------
    input_img : ImageArray
        The input image array before suppression
    window_size :  Optional[int]
        Suppression windows size, only works for odd number, if none are supplied, a default value of 3 is used

    Returns
    -------
    ImageArray
        An ImageArray after suppression
    """
    height, width = np.shape(input_img)
    center_window_index = window_size ** 2 // 2
    input_img = input_img.flatten()

    # Create sliding window that contain correct index from the input image
    window = []
    for i in range(window_size):
        window += [i * width + j for j in range(window_size)]
    window = np.array(window)

    row = 0

    # Move the windows from top left to bottom right until the end
    while window[-1] < len(input_img):
        max_index = np.argmax(input_img[window])

        # suppress non max to 0 if nearest (n*n - 1) neighbours have a higher activation
        if max_index != center_window_index:
            input_img[window[center_window_index]] = 0

        # shift to next row
        if window[0] + window_size > width * row + width - 1:
            window += window_size
            row += 1
        else:
            window += 1

    return input_img.reshape(height, width)

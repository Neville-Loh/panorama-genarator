# The following implementation of Harris Corner Detection is based on Chapter 2 of Solem 2012, see:
# Solem, J. E. (2012). Programming Computer Vision with Python:
# Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".
# The code has been modified to work with updated packages and python 3 vs. the original python 2 implementation.
import numpy as np
from pylab import *
from numpy import *
from scipy.ndimage import gaussian_filter
from PIL import Image
from CS773StitchingSkeleton import filenameToSmoothedAndScaledpxArray
from data_exploration.image_plot import prepareMatchingImage

CHECKER_BOARD = "../../images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "../../images/panoramaStitching/tongariro_left_01.png"
MOUNTAIN_RIGHT = "../../images/panoramaStitching/tongariro_right_01.png"
MOUNTAIN_SMALL_TEST = "../../images/panoramaStitching/tongariro_left_01_small.png"
SNOW_LEFT = "../../images/panoramaStitching/snow_park_left_berg_loh_02.png"
SNOW_RIGHT = "../../images/panoramaStitching/snow_park_right_berg_loh_02.png"
OXFORD_LEFT = "../../images/panoramaStitching/oxford_left_berg_loh_01.png"
OXFORD_RIGHT = "../../images/panoramaStitching/oxford_right_berg_loh_01.png"


def compute_harris_response(im, sigma=3):
    # derivatives
    imx = zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the harris matrix
    Wxx = gaussian_filter(imx * imx, sigma)
    Wxy = gaussian_filter(imx * imy, sigma)
    Wyy = gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating
        corners and image boundary. """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T

    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates (reverse to get descending order)
    index = argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],
         [p[0] for p in filtered_coords], '.')
    axis('off')
    show()


def solemCornerDetection(left_image_location, right_image_location=None, plot=False):
    left_px_array = filenameToSmoothedAndScaledpxArray(left_image_location)
    left_px_numpy_array = np.array(left_px_array)
    left_harrisim = compute_harris_response(left_px_numpy_array)
    left_filtered_coords = get_harris_points(left_harrisim, 6)

    if right_image_location is not None:
        right_px_array = filenameToSmoothedAndScaledpxArray(right_image_location)
        right_px_numpy_array = np.array(right_px_array)
        right_harrisim = compute_harris_response(right_px_numpy_array)
        right_filtered_coords = get_harris_points(right_harrisim, 6)
    if plot:
        height, width = len(left_px_array), len(left_px_array[0])
        if right_image_location is None:
            plot_harris_points(left_px_array, left_filtered_coords)
        else:
            matching_image = prepareMatchingImage(left_px_array, right_px_array, width, height)
            plt.imshow(matching_image, cmap='gray')
            ax = plt.gca()
            ax.set_title("Solem Harris Corner Detection Output")
            plt.plot([p[1] for p in left_filtered_coords],
                     [p[0] for p in left_filtered_coords], '.')

            plt.plot([p[1] + width for p in right_filtered_coords],
                     [p[0] for p in right_filtered_coords], '.')
        plt.show()

    return left_filtered_coords


if __name__ == "__main__":
    solemCornerDetection(MOUNTAIN_LEFT, MOUNTAIN_RIGHT, plot=True)

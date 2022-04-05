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
from data_exploration.image_plot import plot_side_by_side_pairs


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
    Wdet = Wxx*Wyy - Wxy**2
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

    harrisim = compute_harris_response(left_px_numpy_array)
    filtered_coords = get_harris_points(harrisim, 6)
    if plot:
        plot_harris_points(left_px_numpy_array, filtered_coords)

        plot_side_by_side_pairs(left_px_array, right_px_array, pairs, title="Before outlier rejection")
        print(f'len of pairs before = {len(pairs)}')
        pairs = reject_outlier_pairs(pairs, width_offset=width, m=1)
        print(f'len of pairs after = {len(pairs)}')
        plot_side_by_side_pairs(left_px_array, right_px_array, pairs, title="After outlier rejection")



    return filtered_coords


if __name__ == "__main__":
    solemCornerDetection(MOUNTAIN_LEFT, plot=True)

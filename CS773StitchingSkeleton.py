from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import heapq
from data_exploration.histograms import plot_histogram

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth

# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
from imageProcessing.harris import compute_gaussian_averaging, get_square_and_mixed_derivatives, get_image_cornerness, \
    get_all_corner, sobel, bruteforce_non_max_suppression, compute_harris_corner

CHECKER_BOARD = "./images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "./images/panoramaStitching/tongariro_left_01.png"
MOUNTAIN_RIGHT = "./images/panoramaStitching/tongariro_right_01.png"
MOUNTAIN_SMALL_TEST = "./images/panoramaStitching/tongariro_left_01_small.png"
SNOW_LEFT = "./images/panoramaStitching/snow_park_left_berg_loh_02.png"
SNOW_RIGHT = "./images/panoramaStitching/snow_park_right_berg_loh_02.png"
OXFORD_LEFT = "./images/panoramaStitching/oxford_left_berg_loh_01.png"
OXFORD_RIGHT = "./images/panoramaStitching/oxford_right_berg_loh_01.png"


def prepareRGBImageFromIndividualArrays(r_pixel_array, g_pixel_array, b_pixel_array, image_width, image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage


def pixelArrayToSingleList(pixelArray):
    list_of_pixel_values = []
    for row in pixelArray:
        for item in row:
            list_of_pixel_values.append(item)
    return list_of_pixel_values


def filenameToSmoothedAndScaledpxArray(filename):
    (image_width, image_height, px_array_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename)

    start = timer()
    px_array_smoothed = IPSmooth.computeGaussianAveraging3x3(px_array_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    start = timer()
    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_smoothed_scaled = IPPixelOps.scaleTo0And255AndQuantize(px_array_smoothed, image_width, image_height)
    end = timer()
    print("elapsed time  image smoothing: ", end - start)
    return px_array_smoothed_scaled


def extension_compare_alphas():
    left_or_right_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_SMALL_TEST)

    alphasToTest = [0.04, 0.05, 0.06]

    for testAlpha in alphasToTest:
        corners = compute_harris_corner(left_or_right_px_array,
                                        n_corner=500,
                                        alpha=testAlpha,
                                        gaussian_window_size=5,
                                        plot_image=True)

        plot_histogram([c.cornerness for c in corners],
                       "Distribution of Corner Values for alpha={}".format(testAlpha)).show()


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = MOUNTAIN_LEFT
    filename_right_image = MOUNTAIN_RIGHT
    filename_simple_test = MOUNTAIN_SMALL_TEST

    left_or_right_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_SMALL_TEST)

    # Task: Extraction of Harris corners
    # According to lecture compute Harris corner for both images
    # Perform a simple non max suppression in a 3x3 neighbour-hood, and report the 1000 strongest corner per image.

    # compute_harris_corner(px_array_left_original,
    #                       n_corner=5000,
    #                       alpha=0.05,
    #                       gaussian_window_size=5,
    #                       plot_image=True)
    #
    # compute_harris_corner(px_array_right_original,
    #                       n_corner=5000,
    #                       alpha=0.05,
    #                       gaussian_window_size=5,
    #                       plot_image=True)


if __name__ == "__main__":
    extension_compare_alphas()

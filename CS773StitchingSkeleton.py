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


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = MOUNTAIN_LEFT
    filename_right_image = MOUNTAIN_RIGHT

    (image_width, image_height, px_array_left_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(
        filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(
        filename_right_image)

    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    # plot_histogram(pixelArrayToSingleList(px_array_left)).show()
    # plot_histogram(pixelArrayToSingleList(px_array_right)).show()

    # # some simplevisualizations
    #
    # fig1, axs1 = pyplot.subplots(1, 2)
    #
    # axs1[0].set_title('Harris response left overlaid on orig image')
    # axs1[1].set_title('Harris response right overlaid on orig image')
    # axs1[0].imshow(px_array_left, cmap='gray')
    # axs1[1].imshow(px_array_right, cmap='gray')
    #
    # # plot a red point in the center of each image
    # circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    # axs1[0].add_patch(circle)
    #
    # circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    # axs1[1].add_patch(circle)
    #
    # pyplot.show()
    #
    # # a combined image including a red matching line as a connection patch artist (from matplotlib)
    #
    # matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)
    #
    # pyplot.imshow(matchingImage, cmap='gray')
    # ax = pyplot.gca()
    # ax.set_title("Matching image")
    #
    # pointA = (image_width/2, image_height/2)
    # pointB = (3*image_width/2, image_height/2)
    # connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    # ax.add_artist(connection)
    #
    # pyplot.show()

    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # Task: Extraction of Harris corners
    # According to lecture compute Harris corner for both images
    # Perform a simple non max suppression in a 3x3 neighbour-hood, and report the 1000 strongest corner per image.

    compute_harris_corner(px_array_left_original,
                          n_corner=5000,
                          alpha=0.05,
                          gaussian_window_size=3,
                          plot_image=True)

    compute_harris_corner(px_array_right_original,
                          n_corner=5000,
                          alpha=0.05,
                          gaussian_window_size=3,
                          plot_image=True)


if __name__ == "__main__":
    main()

import argparse

import numpy as np
from matplotlib import pyplot
from matplotlib.patches import ConnectionPatch
import sys

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
from data_exploration.histograms import plot_histogram
from data_exploration.util import slope, reject_outliers, reject_pair_outliers
from image_stiching.feature_descriptor.feature_descriptor import get_patches, compare
from image_stiching.harris_conrner_detection.harris import compute_harris_corner

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


def basic_comparison():
    left_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_LEFT)
    right_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_RIGHT)

    height, width = len(left_px_array), len(left_px_array[0])

    left_corners = compute_harris_corner(left_px_array,
                                         n_corner=1000,
                                         alpha=0.04,
                                         gaussian_window_size=3,
                                         plot_image=False)

    right_corners = compute_harris_corner(right_px_array,
                                          n_corner=1000,
                                          alpha=0.04,
                                          gaussian_window_size=5,
                                          plot_image=False)

    left_corners = get_patches(left_corners, 15, left_px_array)
    right_corners = get_patches(right_corners, 15, right_px_array)

    pairs = compare(left_corners, right_corners)

    slops = []
    s = []
    for pair in pairs:
        sl = slope(pair[0].x, pair[0].y, pair[1].x + width, pair[1].y)
        s.append(sl)
        slops.append((pair[0], pair[1], sl))
        print(sl)

    fig1, ax1 = pyplot.subplots(1, 2)
    ax1[0].set_title('Before rejection')
    ax1[0].boxplot(s)
    # pyplot.show()

    s = reject_outliers(np.array(s))

    ax1[1].set_title('After rejection')
    ax1[1].boxplot(s)
    pyplot.show()

    matching_image = prepareMatchingImage(left_px_array, right_px_array, width, height)
    pyplot.imshow(matching_image, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")


    for pair in pairs:
        point_a = (pair[0].x, pair[0].y)
        point_b = (pair[1].x + width, pair[1].y)
        connection = ConnectionPatch(point_a, point_b, "data", edgecolor='r', linewidth=1)
        ax.add_artist(connection)

    pyplot.show()

    matching_image = prepareMatchingImage(left_px_array, right_px_array, width, height)
    pyplot.imshow(matching_image, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image")

    pairs = reject_pair_outliers(slops, s)
    print(len(pairs))
    for pair in pairs:
        point_a = (pair[0].x, pair[0].y)
        point_b = (pair[1].x + width, pair[1].y)
        connection = ConnectionPatch(point_a, point_b, "data", edgecolor='r', linewidth=1)
        ax.add_artist(connection)

    pyplot.show()


def main():
    # Retrieve all command line argument
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    # If there is no argument, compute a basic comparison with default image
    if len(args) == 0 and len(opts) == 0:
        basic_comparison()

    # Parse all additional argument if there is any
    else:
        parser = argparse.ArgumentParser(description='Description of your program')

        # input image path parameters
        parser.add_argument('input', metavar='input', type=str)

        # Corner number argument Optional
        parser.add_argument('-n', '--n_corner',
                            help='Number of corner output by the algorithm. The output image will contain n corners '
                                 'with the strongest response. If nothing is supplied, default to 1000',
                            default=1000)

        # Gaussian windows size argument Optional
        parser.add_argument('-a', '--alpha',
                            help='The Harris Response constant alpha. Specifies the weighting between corner with '
                                 'strong with single direction and multi-direction. A higher alpha will result in '
                                 'less difference between response of ingle direction and multi-direction shift in '
                                 'intensity. If nothing is supplied, default to 0.04'
                            , default=0.04)

        # Gaussian windows size argument, Optional
        parser.add_argument('-w', '--winsize',
                            help='Gaussian windows size which applied the the squared and mix derivative of the image.'
                                 'A higher windows size will result in higher degree of smoothing, If nothing is '
                                 'supplied, the default widows size is set to 5.',
                            default=5)
        args = vars(parser.parse_args())

        # Compute and plot Harris Corner with optional or default values
        img = filenameToSmoothedAndScaledpxArray(args['input'])
        compute_harris_corner(img,
                              n_corner=int(args['n_corner']),
                              alpha=float(args['alpha']),
                              gaussian_window_size=int(args['winsize']),
                              plot_image=True)


if __name__ == "__main__":
    main()

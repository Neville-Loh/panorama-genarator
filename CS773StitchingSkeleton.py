import sys
import argparse
import numpy as np
from matplotlib import pyplot

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.smoothing as IPSmooth
from data_exploration.image_plot import plot_side_by_side_pairs
from data_exploration.util import reject_outliers, reject_pair_outliers
from image_stiching.feature_descriptor.feature_descriptor import get_patches, compare_all_ncc, match_corner_by_ncc, \
    reject_outlier_pairs
from image_stiching.harris_conrner_detection.harris import compute_harris_corner
from image_stiching.performance_evaulation.timer import measure_elapsed_time

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


def pixelArrayToSingleList(pixelArray):
    list_of_pixel_values = []
    for row in pixelArray:
        for item in row:
            list_of_pixel_values.append(item)
    return list_of_pixel_values


@measure_elapsed_time
def filenameToSmoothedAndScaledpxArray(filename):
    (image_width, image_height, px_array_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename)
    px_array_smoothed = IPSmooth.computeGaussianAveraging3x3(px_array_original, image_width, image_height)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_smoothed_scaled = IPPixelOps.scaleTo0And255AndQuantize(px_array_smoothed, image_width, image_height)
    return px_array_smoothed_scaled


def basic_comparison():
    left_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_LEFT)
    right_px_array = filenameToSmoothedAndScaledpxArray(MOUNTAIN_RIGHT)

    height, width = len(left_px_array), len(left_px_array[0])

    left_corners = compute_harris_corner(left_px_array,
                                         n_corner=1000,
                                         alpha=0.04,
                                         gaussian_window_size=7,
                                         plot_image=False)

    right_corners = compute_harris_corner(right_px_array,
                                          n_corner=1000,
                                          alpha=0.04,
                                          gaussian_window_size=7,
                                          plot_image=False)

    # get the best matches for each corner in the left image
    pairs = match_corner_by_ncc((left_px_array, left_corners),
                                (right_px_array, right_corners),
                                feature_descriptor_path_size=15,
                                threshold=0.9)

    slope = [pair.cal_gradient(width_offset=width) for pair in pairs]
    fig1, ax1 = pyplot.subplots(1, 2)
    ax1[0].set_title('Before rejection')
    ax1[0].boxplot(slope)

    s = reject_outliers(np.array(slope))
    ax1[1].set_title('After rejection')
    ax1[1].boxplot(s)
    pyplot.show()

    plot_side_by_side_pairs(left_px_array, right_px_array, pairs, title="Before outlier rejection", unique_color=False)
    print(f'len of pairs before = {len(pairs)}')
    pairs = reject_outlier_pairs(pairs, width_offset=width, m=1)
    print(f'len of pairs after = {len(pairs)}')
    plot_side_by_side_pairs(left_px_array, right_px_array, pairs, title="After outlier rejection", unique_color=True)


def main():
    # Retrieve all command line argument
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    # If there is no argument, compute a basic comparison with default image
    if len(args) == 0 and len(opts) == 0:
        basic_comparison()

    # Parse all additional argument if there is any
    else:
        parser = argparse.ArgumentParser(description='A basic image stitching program written by Neville Loh and '
                                                     'Nicholas Berg.')

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

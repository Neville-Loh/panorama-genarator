import argparse

from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import sys
from data_exploration.histograms import plot_histogram

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
import imageProcessing.naivecornerdetection as naive
import image_stiching.harris_conrner_detection.SolemHarrisImplementation as Solem
from image_stiching.feature_descriptor.feature_descriptor import get_patches, compare
from image_stiching.harris_conrner_detection.harris import compute_harris_corner
from scipy.spatial import distance


def extension_compare_three_corner_algorithms_on_two_or_more_images(images=[MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT]):
    fig1, axs1 = pyplot.subplots(len(images), 3, sharey=True)

    for image in images:
        image_index = images.index(image)
        image_px_array = filenameToSmoothedAndScaledpxArray(image)

        harris_corners_image = compute_harris_corner(image_px_array,
                                                     n_corner=1000,
                                                     alpha=0.04,
                                                     gaussian_window_size=5,
                                                     plot_image=False)
        print("We detected {} harris corners on image {}".format(len(harris_corners_image), image_index))

        solem_corners = Solem.solemCornerDetection(image, False)
        print("We detected {} Solem harris corners on image {}".format(len(solem_corners), image_index))

        naive_corners = naive.naiveDetection(image_px_array, 100, False)
        print("We detected {} naive corners on image {}".format(len(naive_corners), image_index))

        axs1[image_index][0].set_title('Berg and Loh Harris response Overlaid on Image {}'.format(image_index))
        axs1[image_index][1].set_title('Solem Harris response Overlaid on Image {}'.format(image_index))
        axs1[image_index][2].set_title('Naive Harris response Overlaid on Image {}'.format(image_index))

        axs1[image_index][0].imshow(image_px_array, cmap='gray')
        axs1[image_index][1].imshow(image_px_array, cmap='gray')
        axs1[image_index][2].imshow(image_px_array, cmap='gray')

        # plot a red point in the center of each image
        for corner in harris_corners_image:
            circle = Circle((corner.x, corner.y), 2.5, color='r')
            axs1[image_index][0].add_patch(circle)

        for corner in solem_corners:
            circle = Circle((corner[1], corner[0]), 2.5, color='r')
            axs1[image_index][1].add_patch(circle)

        for corner in naive_corners:
            circle = Circle((corner[0], corner[1]), 2.5, color='r')
            axs1[image_index][2].add_patch(circle)

    pyplot.show()


def extension_compare_alphas(alphasToTest=[0.01, 0.05, 0.2], images=[MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT],
                             histogram=True):
    fig1, axs1 = pyplot.subplots(len(images), 3)

    image_corners = []

    for image in images:
        image_index = images.index(image)
        image_px_array = filenameToSmoothedAndScaledpxArray(image)

        alpha_corners = []

        for testAlpha in alphasToTest:

            alpha_index = alphasToTest.index(testAlpha)
            corners = compute_harris_corner(image_px_array,
                                            n_corner=1000,
                                            alpha=testAlpha,
                                            gaussian_window_size=5,
                                            plot_image=False)

            alpha_corners.append([c.cornerness for c in corners])

            axs1[image_index][alpha_index].set_title(
                'Berg and Loh Harris Response with alpha={} Overlaid on Image {}'.format(testAlpha, image_index))
            axs1[image_index][alpha_index].imshow(image_px_array, cmap='gray')

            for corner in corners:
                circle = Circle((corner.x, corner.y), 2.5, color='r')
                axs1[image_index][alpha_index].add_patch(circle)

        image_corners.append(alpha_corners)

    pyplot.show()

    if histogram:
        fig1, axs1 = pyplot.subplots(len(images), len(alphasToTest), sharey=True, tight_layout=True)

        for image in images:
            image_index = images.index(image)

            for testAlpha in alphasToTest:
                alpha_index = alphasToTest.index(testAlpha)
                axs1[image_index][alpha_index].hist(x=image_corners[image_index][alpha_index], bins='auto',
                                                    color='#0504aa', alpha=0.7, rwidth=0.85)
                axs1[image_index][alpha_index].grid(axis='y', alpha=0.75)
                axs1[image_index][alpha_index].set_xlabel("Cornerness")
                axs1[image_index][alpha_index].set_ylabel("Frequency")
                axs1[image_index][alpha_index].set_title(
                    'Distribution of Corners Responses for Alpha={} Overlaid on Image {}'.format(testAlpha,
                                                                                                 image_index))
        pyplot.show()


def extension_compare_window_size(windowsToTest=[3, 5, 7, 9], images=[MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT],
                                  histogram=True):
    fig1, axs1 = pyplot.subplots(len(images), len(windowsToTest))

    image_corners = []

    for image in images:
        image_index = images.index(image)
        image_px_array = filenameToSmoothedAndScaledpxArray(image)
        window_size_corners = []

        for window_size in windowsToTest:

            window_index = windowsToTest.index(window_size)
            corners = compute_harris_corner(image_px_array,
                                            n_corner=1000,
                                            alpha=0.04,
                                            gaussian_window_size=window_size,
                                            plot_image=False)

            window_size_corners.append([c.cornerness for c in corners])

            axs1[image_index][window_index].set_title(
                'Berg and Loh Harris Response Gaussian Window as {}x{} Overlaid on Image {}'.format(window_size,
                                                                                                    window_size,
                                                                                                    image_index))
            axs1[image_index][window_index].imshow(image_px_array, cmap='gray')

            # plot a red point in the center of each image
            for corner in corners:
                circle = Circle((corner.x, corner.y), 2.5, color='r')
                axs1[image_index][window_index].add_patch(circle)

        image_corners.append(window_size_corners)

    pyplot.show()

    if histogram:
        fig1, axs1 = pyplot.subplots(len(images), len(windowsToTest), sharey=True, tight_layout=True)

        for image in images:
            image_index = images.index(image)

            for window_size in windowsToTest:
                window_index = windowsToTest.index(window_size)
                axs1[image_index][window_index].hist(x=image_corners[image_index][window_index], bins='auto',
                                                     color='#0504aa', alpha=0.7, rwidth=0.85)
                axs1[image_index][window_index].grid(axis='y', alpha=0.75)
                axs1[image_index][window_index].set_xlabel("Cornerness")
                axs1[image_index][window_index].set_ylabel("Frequency")
                axs1[image_index][window_index].set_title(
                    'Distribution of Corners Responses for Gaussian Window as {}x{} Overlaid on Image {}'.format(
                        window_size, window_size, image_index))
        pyplot.show()


def extension_distribution_of_distances_between_points(alphasToTest=[0.01, 0.05, 0.2],
                                                       images=[MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT], histogram=True):
    fig1, axs1 = pyplot.subplots(len(images), 3)

    image_corners = []

    for image in images:
        image_index = images.index(image)
        image_px_array = filenameToSmoothedAndScaledpxArray(image)

        alpha_corners = []

        for testAlpha in alphasToTest:

            corner_coordinates = []

            alpha_index = alphasToTest.index(testAlpha)
            corners = compute_harris_corner(image_px_array,
                                            n_corner=1000,
                                            alpha=testAlpha,
                                            gaussian_window_size=5,
                                            plot_image=False)

            axs1[image_index][alpha_index].set_title(
                'Berg and Loh Harris Response with alpha={} Overlaid on Image {}'.format(testAlpha, image_index))
            axs1[image_index][alpha_index].imshow(image_px_array, cmap='gray')

            for corner in corners:
                circle = Circle((corner.x, corner.y), 2.5, color='r')
                axs1[image_index][alpha_index].add_patch(circle)

                corner_coordinates.append((corner.x, corner.y))

            alpha_corners.append(corner_coordinates)

        image_corners.append(alpha_corners)

    pyplot.show()

    if histogram:
        fig1, axs1 = pyplot.subplots(len(images), len(alphasToTest), sharey=True, tight_layout=True)

        for image in images:
            image_index = images.index(image)

            for testAlpha in alphasToTest:
                alpha_index = alphasToTest.index(testAlpha)

                coordinates = np.array(image_corners[image_index][alpha_index])
                distances = distance.pdist(coordinates, 'euclidean')

                axs1[image_index][alpha_index].hist(x=distances, bins='auto',
                                                    color='#0504aa', alpha=0.7, rwidth=0.85)
                axs1[image_index][alpha_index].grid(axis='y', alpha=0.75)
                axs1[image_index][alpha_index].set_xlabel("Distance Between Points")
                axs1[image_index][alpha_index].set_ylabel("Frequency")
                axs1[image_index][alpha_index].set_title(
                    'Distance between Corners for Alpha={} Overlaid on Image {}'.format(testAlpha,
                                                                                        image_index))
        pyplot.show()


def extension_naiveDetection():
    left_or_right_px_array = filenameToSmoothedAndScaledpxArray(OXFORD_LEFT)

    corners = naive.naiveDetection(left_or_right_px_array, 100, True)

    plot_histogram([c[2] for c in corners],
                   "Distribution of Naieve Corner Values").show()

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


#### CODE COMMENTED OUT FROM CS773StitchingSkeleton

    # fig1, axs1 = pyplot.subplots(1, 2)
    # axs1[0].set_title('Harris response left overlaid on orig image')
    # axs1[1].set_title('Harris response right overlaid on orig image')
    # axs1[0].imshow(left_px_array, cmap='gray')
    # axs1[1].imshow(right_px_array, cmap='gray')


    # for corner in left_corners:
    #     circle = Circle((corner.x, corner.y), 3.5, color='r')
    #     axs1[0].add_patch(circle)
    #
    # for corner in right_corners:
    #     circle = Circle((corner.x, corner.y), 3.5, color='r')
    #     axs1[1].add_patch(circle)
    #
    # print("We plotted {} corners on the left image".format(len(left_corners)))
    # print("We plotted {} corners on the right image".format(len(right_corners)))
    #
    # pyplot.show()
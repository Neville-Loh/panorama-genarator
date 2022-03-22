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
import imageProcessing.naivecornerdetection as naive
import imageProcessing.SolemHarrisImplementation as Solem
from imageProcessing.harris import compute_harris_corner

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

    left_corners = compute_harris_corner(left_px_array,
                                         n_corner=1000,
                                         alpha=0.04,
                                         gaussian_window_size=5,
                                         plot_image=False)

    right_corners = compute_harris_corner(right_px_array,
                                         n_corner=1000,
                                         alpha=0.04,
                                         gaussian_window_size=5,
                                         plot_image=False)

    fig1, axs1 = pyplot.subplots(1, 2)
    axs1[0].set_title('Harris response left overlaid on orig image')
    axs1[1].set_title('Harris response right overlaid on orig image')
    axs1[0].imshow(left_px_array, cmap='gray')
    axs1[1].imshow(right_px_array, cmap='gray')

    for corner in left_corners:
        circle = Circle((corner.x, corner.y), 3.5, color='r')
        axs1[0].add_patch(circle)

    for corner in right_corners:
        circle = Circle((corner.x, corner.y), 3.5, color='r')
        axs1[1].add_patch(circle)

    print("We plotted {} corners on the left image".format(len(left_corners)))
    print("We plotted {} corners on the right image".format(len(right_corners)))

    pyplot.show()

def extension_compare_three_corner_algorithms_on_two_or_more_images(images = [MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT]):

    fig1, axs1 = pyplot.subplots(len(images), 3)

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


def extension_compare_alphas(alphasToTest = [0.01, 0.05, 0.2], images = [MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT]):

    fig1, axs1 = pyplot.subplots(len(images), 3)

    for image in images:
        image_index = images.index(image)
        image_px_array = filenameToSmoothedAndScaledpxArray(image)

        for testAlpha in alphasToTest:
            alpha_index = alphasToTest.index(testAlpha)
            corners = compute_harris_corner(image_px_array,
                                            n_corner=1000,
                                            alpha=testAlpha,
                                            gaussian_window_size=5,
                                            plot_image=False)

            axs1[image_index][alpha_index].set_title('Berg and Loh Harris Response with alpha={} Overlaid on Image {}'.format(testAlpha, image_index))
            axs1[image_index][alpha_index].imshow(image_px_array, cmap='gray')

            for corner in corners:
                circle = Circle((corner.x, corner.y), 2.5, color='r')
                axs1[image_index][alpha_index].add_patch(circle)

    pyplot.show()

def extension_compare_window_size(windowsToTest = [3, 5, 7, 9], images = [MOUNTAIN_LEFT, OXFORD_LEFT, SNOW_LEFT], histogram=True):
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
                'Berg and Loh Harris Response Gaussian Window as {}x{} Overlaid on Image {}'.format(window_size, window_size, image_index))
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
                axs1[image_index][window_index].hist(x=image_corners[image_index][window_index], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
                axs1[image_index][window_index].grid(axis='y', alpha=0.75)
                axs1[image_index][window_index].set_xlabel("Cornerness")
                axs1[image_index][window_index].set_ylabel("Frequency")
                axs1[image_index][window_index].set_title('Distribution of Corners Responses for Gaussian Window as {}x{} Overlaid on Image {}'.format(window_size, window_size, image_index))
        pyplot.show()


def extension_thresholds_and_histograms():
    # plot_histogram([c.cornerness for c in corners],
    #                "Distribution of Corner Values for alpha={}".format(testAlpha)).show()
    pass

def extension_naiveDetection():
    left_or_right_px_array = filenameToSmoothedAndScaledpxArray(OXFORD_LEFT)

    corners = naive.naiveDetection(left_or_right_px_array, 100, True)

    plot_histogram([c[2] for c in corners],
                   "Distribution of Naieve Corner Values").show()

def main():
    basic_comparison()
    extension_compare_three_corner_algorithms_on_two_or_more_images()
    extension_compare_alphas()
    extension_compare_window_size()
    extension_thresholds_and_histograms()

if __name__ == "__main__":
    extension_compare_window_size()

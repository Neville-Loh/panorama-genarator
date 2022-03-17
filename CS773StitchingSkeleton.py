from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import heapq

from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth

# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
from imageProcessing.harris import compute_gaussian_averaging, get_square_and_mixed_derivatives, get_image_cornerness, \
    get_all_corner, sobel, bruteforce_non_max_suppression

CHECKER_BOARD = "./images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "./images/panoramaStitching/tongariro_left_01.png"


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


# This is our code skeleton that performs the stitching
def main():
    filename_left_image = MOUNTAIN_LEFT
    filename_right_image = MOUNTAIN_LEFT

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

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    # Task: Extraction of Harris corners
    # According to lecture compute Harris corner for both images
    # Perform a simple non max suppression in a 3x3 neigbour-hood, and report the 1000 strongest corner per image.

    px_array_left_original = np.array(px_array_left_original)
    # px_array_left = np.array(px_array_left)

    # Step 1
    # Use a 3x3 averaging as an initial blurring

    fig1, axs1 = pyplot.subplots(4, 4, figsize=(12, 10))
    axs1[0][0].imshow(px_array_left_original, cmap='gray')
    axs1[0][1].imshow(px_array_left, cmap='gray')

    # Step 2
    # Implement e.g. Sobel filter in x and y, (The gradiate)  for X and Y derivatives

    ix, iy = sobel(px_array_left, image_width, image_height)
    axs1[0][2].imshow(ix, cmap='gray')
    axs1[0][3].imshow(iy, cmap='gray')

    # Step 3
    # compute the square derivatives and the product of the mixed derivatives, smooth them,
    # Play with different size of gaussian window (5x5, 7x7, 9x9)

    ix2, iy2, ixiy = t = get_square_and_mixed_derivatives(ix, iy)
    axs1[1][0].imshow(ix2, cmap='gray')
    axs1[1][1].imshow(iy2, cmap='gray')
    axs1[1][2].imshow(ixiy, cmap='gray')
    axs1[1][3].axis('off')

    # gaussian blur
    ix2_blur, iy2_blur, ixiy_blur = [compute_gaussian_averaging(img, image_width, image_height) for img in t]

    axs1[2][0].imshow(ix2_blur, cmap='gray')
    axs1[2][1].imshow(iy2_blur, cmap='gray')
    axs1[2][2].imshow(ixiy_blur, cmap='gray')
    # axs1[2][3].axis('off')

    # Choose a Harris constant between 0.04 and 0.06

    # 5 extract Harris corner as (x,y) tuples in a data structure, which is sorted according to the strength of the
    # Harris response function C, sorted list of tuples
    corner_img_array = get_image_cornerness(ix2_blur, iy2_blur, ixiy_blur, 0.04)
    axs1[2][3].imshow(corner_img_array, cmap='gray')

    # 5.5 non-max suppression
    corner_img_array = bruteforce_non_max_suppression(corner_img_array)

    # 6 Prepare n=1000 strongest conner per image
    pq_1000_coor = [(corner.y, corner.x) for corner in heapq.nsmallest(1000, get_all_corner(corner_img_array))]

    # some simple visualizations
    axs1[3][0].imshow(px_array_left_original, cmap='gray')
    axs1[3][0].scatter(*zip(*pq_1000_coor), s=1, color='r')

    pyplot.show()


if __name__ == "__main__":
    main()

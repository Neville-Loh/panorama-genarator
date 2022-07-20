# The following implementation of Normalised Cross Correlation continues from Chapter 2 of Solem 2012, see:
# Solem, J. E. (2012). Programming Computer Vision with Python:
# Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".
# The code has been modified to work with updated packages and python 3 vs. the original python 2 implementation.
import numpy as np
from pylab import *
from numpy import argsort
from matplotlib import colors as colors
from matplotlib import cm as cmx

from image_stitching import filenameToSmoothedAndScaledpxArray
from image_stiching.harris_conrner_detection.SolemHarrisImplementation import compute_harris_response, get_harris_points

CHECKER_BOARD = "../images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "../images/panoramaStitching/tongariro_left_01.png"
MOUNTAIN_RIGHT = "../images/panoramaStitching/tongariro_right_01.png"
MOUNTAIN_SMALL_TEST = "../images/panoramaStitching/tongariro_left_01_small.png"
SNOW_LEFT = "../images/panoramaStitching/snow_park_left_berg_loh_02.png"
SNOW_RIGHT = "../images/panoramaStitching/snow_park_right_berg_loh_02.png"
OXFORD_LEFT = "../images/panoramaStitching/oxford_left_berg_loh_01.png"
OXFORD_RIGHT = "../images/panoramaStitching/oxford_right_berg_loh_01.png"


def get_descriptors(image, filtered_coords, wid=5):
    """ For each point return pixel values around the point
        using a neighbourhood of width 2*wid+1. (Assume points are
        extracted with min_distance > wid). """

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid + 1,
                coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.9):
    """ For each corner point descriptor in the first image,
        select its match to second image using
        normalized cross correlation. """

    n = len(desc1[0])

    # pair-wise distances
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_twosided(desc1, desc2, threshold=0.5):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return concatenate((im1, im2), axis=1)


def plot_matches(im1: np.array, im2: np.array, locs1, locs2, matchscores, show_below=False, unique_color: bool = True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
        matchscores (as output from 'match()'),
        show_below (if images should be shown below matches). """

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = vstack((im3, im3))

    imshow(im3)

    if unique_color:
        cmap = plt.cm.jet
        cNorm = colors.Normalize(vmin=0, vmax=(matchscores > 0).sum())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        colorIndex = 0

    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            if unique_color:
                colorVal = scalarMap.to_rgba(colorIndex)
                colorIndex += 1
            else:
                colorVal = 'c'
            plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], colorVal)
    axis('off')


if __name__ == "__main__":
    im1 = np.array(filenameToSmoothedAndScaledpxArray(MOUNTAIN_LEFT))
    im2 = np.array(filenameToSmoothedAndScaledpxArray(MOUNTAIN_RIGHT))

    wid = 5
    harrisim = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim, wid + 1)
    d1 = get_descriptors(im1, filtered_coords1, wid)
    harrisim = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim, wid + 1)
    d2 = get_descriptors(im2, filtered_coords2, wid)
    matches = match_twosided(d1, d2)
    figure()
    gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
    show()

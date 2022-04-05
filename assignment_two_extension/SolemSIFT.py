# The following implementation of Scale-invariant feature transform (SIFT) continues from Chapter 2 of Solem 2012, see:
# Solem, J. E. (2012). Programming Computer Vision with Python:
# Tools and algorithms for analyzing images. " O'Reilly Media, Inc.".
# The code has been modified to work with updated packages and python 3 vs. the original python 2 implementation.
# Due to the complexity of a SIFT implementation, as part of our extension we are relying on existing implemenations for
# comparison with our NCC
import math

from PIL import Image
from pylab import *
from data_exploration.histograms import plot_histogram
import os

LOCAL_MOUNTAIN = "tongariro_left_01.png"
CHECKER_BOARD = "../images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "../images/panoramaStitching/tongariro_left_01.png"
MOUNTAIN_RIGHT = "../images/panoramaStitching/tongariro_right_01.png"
MOUNTAIN_SMALL_TEST = "../images/panoramaStitching/tongariro_left_01_small.png"
SNOW_LEFT = "../images/panoramaStitching/snow_park_left_berg_loh_02.png"
SNOW_RIGHT = "../images/panoramaStitching/snow_park_right_berg_loh_02.png"
OXFORD_LEFT = "../images/panoramaStitching/oxford_left_berg_loh_01.png"
OXFORD_RIGHT = "../images/panoramaStitching/oxford_right_berg_loh_01.png"

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """ Process an image and save the results in a file. """

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("sift " + imagename + " --output=" + resultname +
               " " + params)
    os.system(cmmd)
    print
    'processed', imagename, 'to', resultname


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # feature locations, descriptors


def write_features_to_file(filename, locs, desc):
    """ Save feature location and descriptor to file. """
    savetxt(filename, hstack((locs, desc)))


def plot_features(im, locs, width=0,color='r', circle=False):
    """ Show image with features. input: im (image as array),
        locs (row, col, scale, orientation of each feature). """

    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plot(x+width, y, color, linewidth=2, )

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


def match(desc1, desc2):
    """ For each descriptor in the first image,
        select its match in the second image.
        input: desc1 (descriptors for the first image),
        desc2 (same for second image). """

    desc1 = array([d / linalg.norm(d) for d in desc1])
    desc2 = array([d / linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0]), 'int')
    desc2t = desc2.T  # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i, :], desc2t)  # vector of dot products
        dotprods = 0.9999 * dotprods
        # inverse cosine and sort, return index for features in second image
        indx = argsort(arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


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


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (location of features),
        matchscores (as output from 'match'), show_below (if images should be shown below). """

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = vstack((im3, im3))

    # show image
    imshow(im3)

    # draw lines for matches
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plot([locs1[i][0], locs2[m][0] + cols1], [locs1[i][1], locs2[m][1]], 'c')
    axis('off')

from matplotlib import colors as colors
from matplotlib import cm as cmx

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

    distances = []

    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            if unique_color:
                colorVal = scalarMap.to_rgba(colorIndex)
                colorIndex += 1
            else:
                colorVal = 'c'
            plot([locs1[i][0], locs2[m][0] + cols1], [locs1[i][1], locs2[m][1]], colorVal)
            distances.append(math.sqrt((locs1[i][0]-locs2[m][0])**2+(locs1[i][1]-locs2[m][1])**2))
    axis('off')
    return distances

def match_twosided(desc1, desc2):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12

def plot_SIFT_circles(left_image, right_image):
    left_image_arry = array(Image.open(left_image).convert('L'))
    right_image_arry = array(Image.open(right_image).convert('L'))

    height, width = len(left_image_arry), len(left_image_arry[0])

    process_image(left_image, 'left_image.sift')
    process_image(right_image, 'right_image.sift')

    left1, leftd1 = read_features_from_file('left_image.sift')
    right1, rightd1 = read_features_from_file('right_image.sift')

    im3 = appendimages(left_image_arry, right_image_arry)
    l_join = appendimages(left1, right1)

    figure()
    gray()
    plot_features(im3, l_join, circle=True)
    plot_features(im3, right1, width=width, circle=True)
    show()

def plot_SIFT_mapping(left_image, right_image):
    left_image_arry = array(Image.open(left_image).convert('L'))
    right_image_arry = array(Image.open(right_image).convert('L'))

    process_image(left_image, 'left_image.sift')
    process_image(right_image, 'right_image.sift')

    left_feature_locations, left_descriptors = read_features_from_file('left_image.sift')
    right_feature_locations, right_descriptors = read_features_from_file('right_image.sift')

    matches = match_twosided(left_descriptors, right_descriptors)

    figure()
    gray()
    distances = plot_matches(left_image_arry, right_image_arry, left_feature_locations, right_feature_locations,
                             matches)
    show()

    plot_histogram(distances, "Frequency distribution of distances between corresponding features", "Distance")
    show()




if __name__ == "__main__":
    left_image = MOUNTAIN_LEFT
    right_image = MOUNTAIN_RIGHT
    plot_SIFT_mapping(left_image, right_image)



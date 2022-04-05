# The Berg-Loh implementation of Histogram of Gradients Feature Descriptors
# This is based on work by Dalal & Triggs, 2005, originally presented in
# Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for
# human detection. In 2005 IEEE computer society conference on computer vision
# and pattern recognition (CVPR'05) (Vol. 1, pp. 886-893). Ieee.

from functools import reduce
import numpy as np
import cv2
from pylab import *
from CS773StitchingSkeleton import filenameToSmoothedAndScaledpxArray
from data_exploration.image_plot import prepareMatchingImage

CHECKER_BOARD = "../images/cornerTest/checkerboard.png"
MOUNTAIN_LEFT = "../images/panoramaStitching/tongariro_left_01.png"
MOUNTAIN_RIGHT = "../images/panoramaStitching/tongariro_right_01.png"
MOUNTAIN_SMALL_TEST = "../images/panoramaStitching/tongariro_left_01_small.png"
SNOW_LEFT = "../images/panoramaStitching/snow_park_left_berg_loh_02.png"
SNOW_RIGHT = "../images/panoramaStitching/snow_park_right_berg_loh_02.png"
OXFORD_LEFT = "../images/panoramaStitching/oxford_left_berg_loh_01.png"
OXFORD_RIGHT = "../images/panoramaStitching/oxford_right_berg_loh_01.png"

# Pseudocode for the HOG Detector:
#      1. Gather an input image
#      2. Normalize gamma & colour
#      3. Compute Gradients
#      4. Weighted vote into spaital & orientation cells
#      5. Contrast normalize over overlapping spatial blocks
#      6. Collect HOG's over detection window
#      7. Linear SVM

# Notes:
#    While in their original work Dalal & Triggs used colour, for performance and simplicity reasons we will work in greyscale.


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                magnitude = math.sqrt(sum(i ** 2 for i in block_vector))
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

def hog_of_one_image():
    img = np.array(filenameToSmoothedAndScaledpxArray(MOUNTAIN_LEFT))
    hog = Hog_descriptor(img, cell_size=16, bin_size=16)
    vector, image = hog.extract()
    plt.imshow(image, cmap='gray')
    plt.show()

def match(desc1, desc2, threshold=0.9):
    """ For each corner point descriptor in the first image,
        select its match to second image using
        normalized cross correlation. """

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

def plot_matches(im1: np.array, im2: np.array, locs1, locs2, matchscores, show_below=True, unique_color: bool = True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
        matchscores (as output from 'match()'),
        show_below (if images should be shown below matches). """
    height, width = len(im1), len(im1[0])
    im3 = prepareMatchingImage(im1, im2)
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
    # hog_of_one_image()
    im1 = np.array(filenameToSmoothedAndScaledpxArray(MOUNTAIN_LEFT))
    im2 = np.array(filenameToSmoothedAndScaledpxArray(MOUNTAIN_RIGHT))
    hog1 = Hog_descriptor(im1, cell_size=16, bin_size=16)
    hog2 = Hog_descriptor(im2, cell_size=16, bin_size=16)
    vector1, hogimage1 = hog1.extract()
    vector2, hogimage2 = hog2.extract()

    height, width = len(im1), len(im1[0])
    matching_image = prepareMatchingImage(im1, im2, width, height)
    matching_image2 = prepareMatchingImage(hogimage1, hogimage2, width, height)

    stacked_image = vstack((matching_image, matching_image2))
    imshow(stacked_image)
    show()
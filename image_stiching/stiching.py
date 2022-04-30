import random
from typing import List, Optional, Tuple

import numpy as np

from data_exploration.image_plot import plot_side_by_side_pairs
from image_stiching.corner import Corner
from image_stiching.feature_descriptor.feature_descriptor import match_corner_by_ncc, reject_outlier_pairs
from image_stiching.harris_conrner_detection.harris import compute_harris_corner
from image_stiching.pair import Pair
from image_stiching.performance_evaulation.timer import measure_elapsed_time
import imageIO.readwrite as IORW
from pprint import pprint
import random
from matplotlib import pyplot as plt
from itertools import product, permutations, combinations

from image_stiching.util.save_object import load_object_at_location


@measure_elapsed_time
def stitch(
        left_px_array: List[List[int]],
        right_px_array: List[List[int]],
        n_corner: Optional[int] = 1000,
        alpha: Optional[float] = 0.04,
        gaussian_window_size: Optional[int] = 7,
        plot_harris_corner: Optional[bool] = False,
        feature_descriptor_patch_size: Optional[int] = 15,
        feature_descriptor_threshold: Optional[float] = 0.9,
        enable_outlier_rejection: Optional[bool] = True,
        outlier_rejection_m: Optional[float] = 1,
        plot_result: Optional[bool] = False,
) -> List[Pair]:
    """
    Stitch two images together.

    parameters:
    -----------
    left_px_array: List[List[int]]
        The greyscale pixel array of the left image.
    right_px_array: List[List[int]]
        The greyscale pixel array of the right image.
    n_corner: Optional[int]
        The number of corners to detect in the left image, default is 1000.
    alpha: Optional[float]
        The alpha value for the Harris corner detector.
    gaussian_window_size: Optional[int]
        The size of the gaussian window for the Harris corner detector, default is 7.
    plot_harris_corner: Optional[bool]
        Whether to plot the detected corners.
    feature_descriptor_path_size: Optional[int]
        The size of the path for the feature descriptor, default is 15.
    feature_descriptor_threshold: Optional[float]
        The threshold for the feature descriptor, default is 0.9.
    enable_outlier_rejection: Optional[bool]
        Whether to enable outlier rejection, default is True.
    outlier_rejection_m: Optional[float]
        The standard deviation for the outlier rejection to include, default is 1.
    plot_result: Optional[bool]
        Whether to plot the result, default is False.

    returns:
    --------
    List[Pair]
        The list of pairs of the matched points.
    """

    height, width = len(left_px_array), len(left_px_array[0])

    left_corners = compute_harris_corner(left_px_array,
                                         n_corner=n_corner,
                                         alpha=alpha,
                                         gaussian_window_size=gaussian_window_size,
                                         plot_image=plot_harris_corner)

    right_corners = compute_harris_corner(right_px_array,
                                          n_corner=1000,
                                          alpha=0.04,
                                          gaussian_window_size=7,
                                          plot_image=False)

    # get the best matches for each corner in the left image
    pairs = match_corner_by_ncc((left_px_array, left_corners),
                                (right_px_array, right_corners),
                                feature_descriptor_patch_size=feature_descriptor_patch_size,
                                threshold=feature_descriptor_threshold)
    if enable_outlier_rejection:
        pairs = reject_outlier_pairs(pairs, width_offset=width, m=outlier_rejection_m)
    if plot_result:
        plot_side_by_side_pairs(left_px_array, right_px_array, pairs, title="Result image with side by side comparison",
                                unique_color=False)

    return pairs


def test_homo():
    pairs = list(load_object_at_location("stuff.txt"))
    result = ransac(pairs, 15000, 0.1)
    h = find_homo(result)

    # get color image
    (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b) = IORW.readRGBImageToSeparatePixelArrays(
        "/home/neville/dev/773/a1-stitching-berg-loh/images/panoramaStitching/tongariro_left_01.png")
    rgb_left_image = np.dstack([pixel_array_r, pixel_array_g, pixel_array_b])

    (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b) = IORW.readRGBImageToSeparatePixelArrays(
        "/home/neville/dev/773/a1-stitching-berg-loh/images/panoramaStitching/tongariro_right_01.png")
    rgb_right_image = np.dstack([pixel_array_r, pixel_array_g, pixel_array_b])

    # create new canvas
    warped_image = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)

    # loop through x and y of a new canvas twice the width of the original image
    for x in range(image_width * 2):
        for y in range(image_height):
            mapped_point = compute_map_point(x, y, h)
            # if source point is in left image

            if x < image_width:
                # mapped point is within left image
                if within(mapped_point, image_height, image_width):
                    # Blend color value from left and interpolated value from right image
                    r, g, b = interpolation_pixel(mapped_point[0], mapped_point[1], rgb_right_image)
                    r_left, g_left, b_left = rgb_left_image[y][x]

                    if r == -1:
                        warped_image[y][x] = rgb_left_image[y][x]
                        continue

                    threshold = 50
                    if r_left - r > threshold or g_left - g > threshold or b_left - b > threshold:
                        warped_image[y][x] = [r, g, b]
                    else:
                        warped_image[y][x] = [(r + r_left) / 2, (g + g_left) / 2, (b + b_left) / 2]

                else:
                    warped_image[y][x] = rgb_left_image[y][x]  # take left pixel

            else:  # source point is outside left image
                if within(mapped_point, image_height, image_width):
                    # take right pixel
                    r, g, b = interpolation_pixel(mapped_point[0], mapped_point[1], rgb_right_image)
                    warped_image[y][x] = [r, g, b]

                    if r == -1:
                        warped_image[y][x] = rgb_right_image[y][x]

    plt.imshow(warped_image)
    plt.show()


def interpolation_pixel(x: float, y: float, rgb_source_image: np.ndarray):
    # get the pixel value from the source image
    image_width, image_height = len(rgb_source_image[0]), len(rgb_source_image)
    r = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 0], image_width, image_height)
    g = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 1], image_width, image_height)
    b = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 2], image_width, image_height)
    return r, g, b


def within(point, image_height, image_width):
    return 0 <= point[0] < image_width and 0 <= point[1] < image_height


def find_homo(pairs: List[Pair]):
    matrix = []
    for pair in pairs:
        x1, y1, x2, y2 = pair.corner1.x, pair.corner1.y, pair.corner2.x, pair.corner2.y
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])

    matrix = np.array(matrix)

    [u, s, vt] = np.linalg.svd(matrix)
    homography = vt[-1].reshape(3, 3)
    # Normalization
    homography = (1 / homography[-1, -1]) * homography
    return homography


def ransac(pairs: List[Pair], iteration: int, threshold: float):
    result = []

    while iteration > 0:
        sample = random.sample(pairs, 4)
        sample_left = [pair.corner1 for pair in sample]
        sample_right = [pair.corner2 for pair in sample]

        if any([points_are_collinear(corners) for corners in combinations(sample_left, 3)]) or \
                any([points_are_collinear(corners) for corners in combinations(sample_right, 3)]):
            continue

        h = find_homo(sample)
        current_result = compute_inliers(h, pairs, threshold)
        result = current_result if len(current_result) > len(result) else result
        iteration -= 1

    return result


def compute_map_point(x, y, homography):
    p_prime = np.dot(homography, np.array([x, y, 1]))
    return (1 / p_prime[-1]) * p_prime


def compute_inliers(homography, pairs, threshold):
    inliers = []
    for pair in pairs:
        p1 = np.array([pair.corner1.x, pair.corner1.y, 1])
        p2 = np.array([pair.corner2.x, pair.corner2.y, 1])
        p2_prime = np.dot(homography, p1)

        # normalization
        p2_prime = (1 / p2_prime[-1]) * p2_prime

        # compute distance between 2 points
        distance = np.sqrt((p2_prime[0] - p2[0]) ** 2 + (p2_prime[1] - p2[1]) ** 2)

        if distance < threshold:
            inliers.append(pair)
    return inliers


def compute_bilinear_interpolation(location_x, location_y, pixel_array, image_width, image_height):
    if location_x < 0 or location_y < 0 or location_x > image_width - 1 or location_y > image_height - 1:
        return -1

    interpolated_value = 0.0

    x = int(location_x)
    y = int(location_y)
    a = location_x - x
    b = location_y - y

    interpolated_value += (1.0 - a) * (1.0 - b) * pixel_array[y][x]
    interpolated_value += a * b * pixel_array[min(y + 1, image_height - 1)][min(x + 1, image_width - 1)]
    interpolated_value += (1.0 - a) * b * pixel_array[min(y + 1, image_height - 1)][x]
    interpolated_value += a * (1.0 - b) * pixel_array[y][min(x + 1, image_width - 1)]

    return interpolated_value


def points_are_collinear(corners: Tuple[Corner, Corner, Corner]):
    p1, p2, p3 = corners
    # area_of_triangle = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    area_of_triangle = 0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    if area_of_triangle < 1e-5:
        print("collinear")
        return True
    else:
        return False


L = [1, 2, 3, 4]
for a in combinations(L, 3):
    print(a)

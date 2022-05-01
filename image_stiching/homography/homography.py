from typing import List, Optional, Tuple

import numpy as np

from image_stiching.corner import Corner
from image_stiching.pair import Pair
import imageIO.readwrite as IORW
import random
from itertools import combinations

from image_stiching.performance_evaulation.timer import measure_elapsed_time

"""
This module contains the homography computation and the RANSAC fitting of the program.
@Author: Neville Loh
"""


@measure_elapsed_time
def fit_transform_homography(pairs: List[Pair],
                             ransac_iteration: Optional[int] = 20000,
                             ransac_threshold: Optional[float] = 1.0,
                             source_left_image_path: Optional[str] = None,
                             source_right_image_path: Optional[str] = None) \
        -> np.ndarray:
    result = ransac(pairs, ransac_iteration, ransac_threshold)
    h = compute_homography(result)

    # read source images
    (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b) \
        = IORW.readRGBImageToSeparatePixelArrays(source_left_image_path)
    rgb_left_image = np.dstack([pixel_array_r, pixel_array_g, pixel_array_b])

    (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b) = \
        IORW.readRGBImageToSeparatePixelArrays(source_right_image_path)
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
                        warped_image[y][x] = [0, 0, 0]

    return warped_image


def interpolation_pixel(x: float, y: float, rgb_source_image: np.ndarray):
    # get the pixel value from the source image
    image_width, image_height = len(rgb_source_image[0]), len(rgb_source_image)
    r = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 0], image_width, image_height)
    g = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 1], image_width, image_height)
    b = compute_bilinear_interpolation(x, y, rgb_source_image[:, :, 2], image_width, image_height)
    return r, g, b


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


def within(point: np.ndarray, image_height: int, image_width: int) -> bool:
    """
    Check if a point is within the image
    Parameters
    ----------
    point: np.ndarray
        point to check
    image_height: int
        height of the image
    image_width: int
        width of the image
    Returns
    -------
    bool
        True if point is within the image, False otherwise
    """
    return 0 <= point[0] < image_width and 0 <= point[1] < image_height


def compute_homography(pairs: List[Pair]) -> np.ndarray:
    """
    Compute the homography matrix
    Parameters
    ----------
    pairs: List[Pair]
        list of pairs of points
    Returns
    -------
    np.ndarray
        homography matrix
    """
    matrix = []
    for pair in pairs:
        x1, y1, x2, y2 = pair.corner1.x, pair.corner1.y, pair.corner2.x, pair.corner2.y
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])

    [_, _, vt] = np.linalg.svd(np.array(matrix))
    vt = vt[-1].reshape(3, 3)
    # Normalization
    homography = (1 / vt[-1, -1]) * vt
    return homography


@measure_elapsed_time
def ransac(pairs: List[Pair], iteration: int, threshold: float) -> List[Pair]:
    """
    RANSAC algorithm
    Parameters
    ----------
    pairs: List[Pair]
        list of pairs of points
    iteration: int
        number of iterations
    threshold: float
        threshold for inliers
    Returns
    -------
    List[Pair]
        list of pairs of points that are inliers
    """
    result = []
    while iteration > 0:
        sample = random.sample(pairs, 4)
        sample_left = [pair.corner1 for pair in sample]
        sample_right = [pair.corner2 for pair in sample]

        if any([points_are_collinear(corners) for corners in combinations(sample_left, 3)]) or \
                any([points_are_collinear(corners) for corners in combinations(sample_right, 3)]):
            continue

        h = compute_homography(sample)
        current_result = compute_inliers(h, pairs, threshold)
        result = current_result if len(current_result) > len(result) else result
        iteration -= 1

    return result


def compute_map_point(x: int, y: int, homography: np.ndarray) -> np.ndarray:
    """
    Compute the map point
    Parameters
    ----------
    x: int
        x coordinate
    y: int
        y coordinate
    homography: np.ndarray
        homography matrix
    Returns
    -------
    np.ndarray
        the matrix that contains the map point
    """
    p_prime = np.dot(homography, np.array([x, y, 1]))
    return (1 / p_prime[-1]) * p_prime


def compute_inliers(homography: np.ndarray, pairs: List[Pair], threshold: float) -> List[Pair]:
    """
    Compute the inliers
    Parameters
    ----------
    homography: np.ndarray
        homography matrix
    pairs: List[Pair]
        list of pairs of points
    threshold: float
        threshold for inliers
    Returns
    -------
    List[Pair]
        list of pairs of points that are inliers
    """
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


def points_are_collinear(corners: Tuple[Corner, Corner, Corner]) -> bool:
    """
    Check if the points are collinear
    Parameters
    ----------
    corners: Tuple[Corner, Corner, Corner]
        tuple of corners
    Returns
    -------
    bool
        True if the points are collinear, False otherwise
    """
    p1, p2, p3 = corners
    area_of_triangle = 0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
    return area_of_triangle < 1e-5

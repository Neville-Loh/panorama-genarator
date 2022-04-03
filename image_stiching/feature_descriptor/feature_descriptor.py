from typing import List, Type, Tuple, Union
import numpy as np
from image_stiching.corner import Corner
from image_stiching.harris_conrner_detection.harris_util import compute_gaussian_averaging
from image_stiching.performance_evaulation.util import measure_elapsed_time

"""
Feature Descriptor
"""


def get_patches(corners: List[Type[Corner]], patch_size: int, img: np.ndarray) -> \
        List[Type[Corner]]:
    """
    Get the patches from the image
    The patch is the region of interest around the corner, which is used for the feature descriptor.
    The patch is a square of size patch_size x patch_size. If a contour is too close to the border,
    the corner is not considered.

    Parameters
    ----------
    corners : List[Type[Corner]]
        List of corners that is outputted by the harris corner detection
    patch_size : int
        Size of the patch of normalized cross correlation
    img : np.ndarray
        Image that is used to get the patches

    Returns
    -------
        List[Type[Corner]] : List of corners with the patches
    """
    center_index = patch_size // 2

    img = np.array(img)
    img = compute_gaussian_averaging(img, windows_size=7)

    result_corners = []
    height, width = img.shape
    for c in corners:
        # ignore border
        if not (c.x < center_index or c.x >= width - center_index
                or c.y < center_index or c.y >= height - center_index):

            # getting the window
            patch: np.ndarray = img[c.y - center_index: c.y + center_index + 1,
                                c.x - center_index: c.x + center_index + 1]

            patch = (patch - np.mean(patch))

            # setting the result
            c.feature_descriptor = patch

            if patch.shape != (15, 15):
                print(f'x = {c.x},y={c.y}, shape={patch.shape}')
            else:
                result_corners.append(c)
            # c.feature_descriptor = patch.flatten()

    return result_corners


def compute_ncc(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
    Compute the normalised cross correlation between two patches.

    parameters
    ----------
    patch1 : np.ndarray
        the patch of the corner from the first image
    patch2 : np.ndarray
        the patch of the corner from the second image

    Returns
    -------
        float, the normalised cross correlation. The higher the value the better it correlates.
    """
    # compute the normalised cross correlation
    return np.sum(patch1 * patch2) / (np.sqrt(np.sum(patch1 ** 2)) * np.sqrt(np.sum(patch2 ** 2)))


def compare(corners1: List[Type[Corner]], corners2: List[Type[Corner]]) -> \
        List[Tuple[Type[Corner], Type[Corner], float]]:
    """
    compare the two list of corners, and return the best matches.

    Parameters
    ----------
    corners1 : List[Type[Corner]]
        List of corners retrieved from the first image
    corners2 : List[Type[Corner]]
        List of corners retrieved from the second image

    Returns
    -------
        List[Type[Corner]]
            List of tuples of the corners that are the best match for each corner in the first list
    """
    pairs = []
    for c1 in corners1:
        # initialize the best match
        first_corner = corners2[0]
        ncc = compute_ncc(c1.feature_descriptor, first_corner.feature_descriptor)
        best = (first_corner, ncc)
        best2 = (first_corner, ncc)
        for c2 in corners2[1:]:
            result = compute_ncc(c1.feature_descriptor, c2.feature_descriptor)

            # check if result greater than 2nd best, if yes, replace 2nd best
            if result > best[1]:
                best2 = best
                best = (c2, result)
            # compare the with the second best value
            elif result > best2[1]:
                best2 = (c2, result)

        # check ratio between 2nd best match and best
        ratio = best2[1] / best[1]
        if ratio <= 0.9:
            pairs.append((c1, best[0], best[1]))

    return pairs

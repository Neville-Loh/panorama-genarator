from typing import List, Type
import numpy as np
from image_stiching.corner import Corner


def get_patches(corners: List[Type[Corner]], patch_size: int, width: int, length: int, img: np.ndarray):
    center_index = patch_size // 2
    img = np.array(img)
    result_corners = []
    print(f'image shpae = {img.shape}')
    for c in corners:
        # ignore border
        if c.x < center_index and c.x > width - center_index \
                 and c.y < center_index and c.y > length - center_index:
            pass
        else:
            #print(type(c.x- center_index))
            # getting the window
            patch:  np.ndarray = img[c.y - center_index: c.y + center_index + 1, c.x - center_index: c.x + center_index + 1]

            patch = (patch - np.mean(patch))

            # setting the result
            c.feature_descriptor = patch


            if patch.shape != (15, 15):
                print(f'x = {c.x},y={c.y}, shape={patch.shape}')
            else:
                result_corners.append(c)
            #c.feature_descriptor = patch.flatten()

    return result_corners
        #print(c, c.feature_descriptor)


def compute_NCC(patch1, patch2):
    """
    Compute the normalised cross correlation between two patches
    """
    # compute the normalised cross correlation
    return np.sum(patch1 * patch2) / (np.sqrt(np.sum(patch1 ** 2)) * np.sqrt(np.sum(patch2 ** 2)))

def compare( corners1: List[Type[Corner]],corners2: List[Type[Corner]]):
    pairs = []
    for c1 in corners1:
        best = (corners2[0], -999999999) # 10
        best2 = (corners2[0], -999999999) # 5

        for c2 in corners2:
            result = compute_NCC(c1.feature_descriptor, c2.feature_descriptor)

            # check if result greater than 2nd best, if yes, replace 2nd best
            if result > best[1]:
                best2 = best
                best = (c2, result)


        # check ratio between 2nd best match and best
        ratio = best2[1]/best[1]
        if ratio <= 0.9:
            pairs.append((c1, best[0], best[1]))

    return pairs
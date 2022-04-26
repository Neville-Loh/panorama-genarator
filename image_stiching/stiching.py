import random
from typing import List, Optional

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
    #pairs = generate_pairs_array()

    pairs = list(load_object_at_location("stuff.txt"))
    #print(pairs)
    random.seed(10)
    result = ransac(pairs, 1500, 1)
    np.set_printoptions(suppress=True)
    print([[(p.corner1.x, p.corner1.y), (p.corner2.x, p.corner2.y)] for p in result])
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
                # print("mapped point: ", mapped_point, x, y)
                # mapped point is outside left image
                if is_outside(mapped_point, image_height, image_width):
                    # take left pixel
                    warped_image[y][x] = rgb_left_image[y][x]

                elif is_inside(mapped_point, image_height, image_width):
                    # Blend color value from left and interpolated value from right image
                    r_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 0],
                                                                                   image_width, image_height) / 2
                    g_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 1],
                                                                                   image_width, image_height) / 2
                    b_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 2],
                                                                                   image_width, image_height) / 2

                    r_left = rgb_left_image[:, :, 0][y][x] / 2
                    g_left = rgb_left_image[:, :, 1][y][x] / 2
                    b_left = rgb_left_image[:, :, 2][y][x] / 2

                    warped_image[y][x][0] = r_bilinear_interpolation + r_left
                    warped_image[y][x][1] = g_bilinear_interpolation + g_left
                    warped_image[y][x][2] = b_bilinear_interpolation + b_left
            else:  # source point is outside left image
                if is_outside(mapped_point, image_height, image_width):
                    # take black pixel
                    warped_image[y][x] = [0, 0, 0]
                elif is_inside(mapped_point, image_height, image_width):
                    # take right pixel
                    r_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 0],
                                                                                   image_width, image_height)
                    g_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 1],
                                                                                   image_width, image_height)
                    b_bilinear_interpolation = compute_bilinear_interpolation(mapped_point[0], mapped_point[1],
                                                                                   rgb_right_image[:, :, 2],
                                                                                   image_width, image_height)

                    warped_image[y][x][0] = r_bilinear_interpolation
                    warped_image[y][x][1] = g_bilinear_interpolation
                    warped_image[y][x][2] = b_bilinear_interpolation

                    ## take pixel from right image
    plt.imshow(warped_image)
    plt.show()

# check if point is outside image
def is_outside(point, image_height, image_width):
    if point[0] < 0 or point[0] >= image_width or point[1] < 0 or point[1] >= image_height:
        return True
    return False

def is_inside(point, image_height, image_width):
    if point[0] >= 0 and point[0] < image_width and point[1] >= 0 and point[1] < image_height:
        return True
    return False

def find_homo(pairs: List[Pair]):
    # pairs = [
    #     Pair(Corner((37, 33), 0), Corner((36, 10), 0), 0),
    #     Pair(Corner((54, 67), 0), Corner((53, 39), 0), 0),
    #     Pair(Corner((56, 56), 0), Corner((56, 27), 0), 0),
    #     Pair(Corner((73, 58), 0), Corner((73, 29), 0), 0),
    # ]
    # print(pairs[0].corner1.x, pairs[0].corner1.y)
    matrix = []
    for pair in pairs:
        x1, y1, x2, y2 = pair.corner1.x, pair.corner1.y, pair.corner2.x, pair.corner2.y
        # print(x1, y1, x2, y2)
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])

    matrix = np.array(matrix)
    pprint(matrix)

    [u, s, vt] = np.linalg.svd(matrix)
    homography = vt[-1].reshape(3, 3)

    # print(homography)
    # print(homography[-1, -1])
    # Normalization
    homography = (1 / homography[-1, -1]) * homography
    return homography


def ransac(pairs: List[Pair], iteration: int, threshold: float):
    result = []
    for _ in range(iteration):
        sample = random.sample(pairs, 4)
        h = find_homo(sample)
        current_result = compute_inliers(h, pairs, threshold)

        result = current_result if len(current_result) > len(result) else result
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


def generate_pairs_array():
    coor = [[(631, 248), (669, 37)], [(354, 369), (361, 187)], [(651, 230), (693, 13)], [(146, 817), (167, 634)],
            [(143, 816), (164, 633)], [(627, 249), (664, 39)], [(685, 330), (720, 134)], [(626, 230), (666, 14)],
            [(627, 334), (656, 143)], [(697, 306), (736, 103)], [(319, 887), (328, 695)], [(634, 349), (663, 160)],
            [(695, 244), (742, 26)], [(693, 393), (721, 208)], [(410, 929), (411, 731)], [(545, 280), (571, 79)],
            [(691, 231), (738, 10)], [(280, 845), (292, 659)], [(412, 219), (426, 8)], [(623, 287), (656, 85)],
            [(682, 301), (720, 98)], [(624, 191), (409, 823)], [(708, 480), (727, 305)], [(389, 266), (400, 65)],
            [(685, 67), (409, 43)], [(666, 310), (701, 112)], [(661, 227), (705, 8)], [(333, 430), (339, 256)],
            [(679, 367), (709, 179)], [(113, 895), (139, 700)], [(707, 373), (739, 183)], [(328, 946), (337, 744)],
            [(333, 885), (341, 693)], [(691, 360), (723, 169)], [(627, 338), (656, 148)], [(148, 820), (169, 636)],
            [(235, 946), (253, 743)], [(568, 309), (594, 114)], [(247, 805), (260, 624)], [(549, 251), (577, 43)],
            [(250, 794), (262, 614)], [(560, 194), (589, 104)], [(611, 238), (648, 25)], [(282, 840), (293, 655)],
            [(256, 788), (268, 609)], [(663, 311), (698, 113)], [(332, 973), (340, 766)], [(137, 830), (159, 645)],
            [(310, 669), (317, 501)], [(699, 296), (740, 90)], [(119, 981), (149, 771)], [(140, 972), (168, 764)],
            [(319, 749), (327, 575)], [(223, 775), (237, 597)], [(702, 291), (678, 239)], [(593, 223), (629, 7)],
            [(578, 252), (610, 44)], [(278, 804), (289, 624)], [(603, 235), (640, 21)], [(537, 261), (564, 57)],
            [(695, 167), (211, 784)], [(386, 890), (389, 698)], [(258, 949), (274, 746)], [(597, 256), (630, 49)],
            [(543, 293), (568, 96)], [(469, 11), (360, 789)], [(627, 183), (421, 949)], [(697, 335), (734, 39)],
            [(609, 74), (360, 789)], [(327, 895), (335, 702)], [(367, 431), (374, 258)], [(693, 382), (722, 195)],
            [(707, 118), (251, 912)], [(654, 413), (679, 233)], [(595, 278), (627, 76)], [(266, 829), (278, 644)],
            [(340, 339), (346, 153)], [(659, 84), (666, 14)], [(335, 916), (343, 719)], [(640, 38), (147, 699)],
            [(337, 926), (345, 728)], [(562, 158), (409, 823)], [(604, 231), (641, 16)], [(254, 743), (265, 569)],
            [(707, 121), (321, 687)], [(382, 399), (390, 222)], [(609, 234), (646, 20)], [(337, 907), (344, 712)],
            [(665, 342), (697, 149)], [(639, 294), (674, 93)], [(529, 49), (578, 45)], [(686, 399), (713, 216)],
            [(672, 237), (716, 20)], [(383, 307), (393, 115)], [(723, 165), (718, 160)], [(420, 222), (435, 12)],
            [(612, 235), (650, 21)], [(135, 843), (157, 656)], [(668, 360), (699, 171)], [(361, 364), (368, 182)],
            [(586, 273), (617, 70)], [(566, 306), (593, 110)], [(119, 895), (145, 699)], [(553, 270), (581, 67)],
            [(330, 939), (338, 738)], [(653, 249), (694, 37)], [(615, 258), (650, 51)], [(700, 481), (719, 306)],
            [(654, 418), (678, 239)], [(600, 337), (627, 147)], [(609, 335), (637, 144)], [(577, 241), (609, 30)],
            [(699, 321), (737, 121)], [(695, 399), (723, 215)], [(693, 401), (720, 216)], [(176, 803), (193, 622)],
            [(392, 324), (402, 135)], [(175, 839), (194, 653)], [(690, 261), (734, 48)], [(623, 283), (657, 80)],
            [(346, 351), (352, 167)], [(615, 201), (262, 614)], [(349, 872), (355, 682)], [(390, 284), (401, 87)],
            [(551, 287), (578, 88)], [(354, 374), (361, 193)], [(699, 145), (164, 633)], [(694, 384), (724, 197)],
            [(641, 226), (683, 8)], [(103, 980), (135, 770)], [(701, 108), (666, 10)], [(269, 936), (283, 736)],
            [(566, 264), (595, 59)], [(669, 354), (700, 164)], [(311, 922), (321, 724)], [(739, 191), (230, 918)],
            [(706, 470), (726, 294)], [(335, 919), (343, 722)], [(251, 781), (263, 603)], [(709, 475), (729, 299)],
            [(335, 265), (340, 64)], [(121, 864), (145, 674)], [(332, 559), (339, 393)], [(364, 574), (370, 408)],
            [(400, 944), (401, 743)], [(624, 294), (657, 94)], [(139, 892), (163, 698)], [(672, 383), (701, 197)],
            [(106, 886), (133, 692)], [(663, 163), (707, 189)], [(641, 401), (665, 220)], [(384, 290), (394, 93)],
            [(714, 435), (739, 254)], [(326, 939), (334, 738)], [(714, 405), (742, 220)], [(151, 936), (176, 734)],
            [(385, 914), (388, 718)], [(595, 263), (628, 57)], [(597, 132), (405, 41)], [(142, 820), (163, 636)],
            [(403, 926), (404, 728)], [(347, 521), (353, 354)], [(555, 261), (584, 56)], [(687, 485), (706, 311)],
            [(338, 978), (345, 770)], [(659, 195), (259, 254)], [(558, 69), (348, 174)], [(693, 286), (734, 79)],
            [(673, 231), (718, 12)], [(332, 396), (338, 218)], [(639, 297), (673, 97)], [(136, 977), (164, 768)],
            [(605, 240), (641, 27)], [(319, 482), (325, 312)], [(270, 830), (282, 646)], [(525, 109), (377, 855)],
            [(642, 244), (682, 31)], [(406, 944), (407, 743)], [(699, 378), (729, 190)], [(235, 962), (253, 756)],
            [(575, 247), (607, 38)], [(605, 246), (641, 36)], [(101, 946), (132, 742)], [(336, 991), (344, 780)],
            [(335, 937), (343, 736)], [(137, 957), (164, 752)], [(516, 258), (541, 53)], [(613, 336), (641, 145)],
            [(709, 541), (721, 370)], [(701, 436), (725, 256)], [(682, 196), (707, 174)], [(616, 333), (645, 141)],
            [(137, 963), (165, 756)], [(377, 446), (384, 275)], [(310, 896), (320, 702)], [(546, 294), (572, 97)],
            [(555, 284), (582, 85)], [(688, 95), (318, 692)], [(101, 949), (132, 744)], [(691, 257), (735, 43)],
            [(323, 911), (332, 715)], [(143, 891), (167, 697)], [(699, 339), (318, 594)], [(639, 287), (237, 597)],
            [(558, 266), (586, 62)], [(672, 404), (698, 222)], [(689, 394), (717, 208)], [(141, 829), (162, 644)],
            [(543, 254), (571, 47)], [(110, 911), (138, 713)], [(290, 918), (302, 720)], [(335, 891), (343, 698)],
            [(141, 894), (165, 700)], [(660, 355), (690, 165)], [(676, 363), (707, 174)], [(307, 658), (315, 490)],
            [(144, 956), (171, 751)], [(701, 305), (741, 101)], [(616, 224), (655, 7)], [(558, 298), (585, 101)],
            [(660, 314), (719, 306)], [(634, 273), (670, 68)], [(542, 238), (571, 27)], [(567, 329), (591, 139)],
            [(193, 705), (205, 533)], [(211, 813), (226, 631)], [(112, 988), (144, 776)], [(593, 243), (627, 32)],
            [(693, 311), (731, 110)], [(394, 919), (396, 722)], [(378, 412), (386, 236)], [(118, 909), (145, 712)],
            [(705, 419), (731, 236)], [(599, 224), (636, 8)], [(657, 63), (399, 900)], [(574, 309), (601, 114)],
            [(534, 284), (559, 84)], [(697, 139), (719, 306)], [(252, 842), (266, 657)], [(331, 271), (336, 71)],
            [(154, 905), (177, 709)], [(391, 249), (402, 45)], [(328, 910), (337, 715)], [(653, 268), (692, 60)],
            [(718, 73), (321, 687)], [(334, 876), (341, 686)], [(679, 363), (710, 173)], [(697, 182), (226, 631)],
            [(540, 289), (565, 91)], [(374, 361), (383, 178)], [(626, 283), (661, 81)], [(633, 129), (227, 313)],
            [(386, 940), (389, 739)], [(618, 112), (610, 44)], [(151, 913), (175, 715)], [(318, 922), (327, 724)],
            [(123, 870), (148, 679)], [(634, 135), (396, 887)], [(132, 949), (159, 746)], [(370, 352), (378, 167)],
            [(330, 274), (335, 75)], [(621, 231), (660, 16)], [(97, 985), (130, 773)], [(336, 973), (344, 767)],
            [(738, 396), (173, 712)], [(178, 868), (198, 678)], [(700, 329), (737, 131)], [(158, 814), (178, 631)],
            [(707, 417), (733, 234)], [(620, 233), (658, 18)]]
    pairs = []
    for p1, p2 in coor:
        pairs.append(Pair(Corner((p1[0], p1[1]), 0), Corner((p2[0], p2[1]), 0), 0))
    return pairs


def compute_bilinear_interpolation(location_x, location_y, pixel_array, image_width, image_height):
    if location_x < 0 or location_y < 0 or location_x > image_width - 1 or location_y > image_height - 1:
        return -1.0

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

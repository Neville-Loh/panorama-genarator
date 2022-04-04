from typing import List
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import imageProcessing.utilities as IPUtils


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage

def plot_side_by_side_pairs(left_px_array: List, right_px_array: List, pairs, title: str = None) -> None:
    """
    Plot an image.

    :param figname: name of the figure
    :param image: image to plot
    :param title: title of the figure
    :param cmap: color map
    :return: None
    """
    height, width = len(left_px_array), len(left_px_array[0])
    matching_image = prepareMatchingImage(left_px_array, right_px_array, width, height)
    plt.imshow(matching_image, cmap='gray')
    ax = plt.gca()
    ax.set_title(title)

    for pair in pairs:
        point_a = (pair[0].x, pair[0].y)
        point_b = (pair[1].x + width, pair[1].y)
        connection = ConnectionPatch(point_a, point_b, "data", edgecolor='r', linewidth=1)
        ax.add_artist(connection)

    plt.show()

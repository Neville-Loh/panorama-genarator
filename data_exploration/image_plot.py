from typing import List, Type
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
from matplotlib.patches import ConnectionPatch
import imageProcessing.utilities as IPUtils

# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
from image_stiching.pair import Pair


def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage


def plot_side_by_side_pairs(left_px_array: List, right_px_array: List, pairs: List[Type[Pair]], title: str = None, unique_color: bool = True) \
        -> None:
    """ Plot a side by side image
    Parameters
    ----------
    left_px_array : List
        The left image pixel array
    right_px_array : List
        The right image pixel array
    pairs : List[Type[Pair]]
        The list of pairs to plot
    title : str
        The title of the plot
    unique_color : bool
        Toggle unique colours for lines or single colours
    """
    height, width = len(left_px_array), len(left_px_array[0])
    matching_image = prepareMatchingImage(left_px_array, right_px_array, width, height)
    plt.imshow(matching_image, cmap='gray')
    ax = plt.gca()
    ax.set_title(title)

    if unique_color:
        cmap = plt.cm.jet
        cNorm = colors.Normalize(vmin=0, vmax=len(pairs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    colorIndex = 0
    for pair in pairs:
        if unique_color:
            colorVal = scalarMap.to_rgba(colorIndex)
        else:
            colorVal = 'r'
        colorIndex += 1
        point_a = (pair.corner1.x, pair.corner1.y)
        point_b = (pair.corner2.x + width, pair.corner2.y)
        connection = ConnectionPatch(point_a, point_b, "data", edgecolor=colorVal, linewidth=1)
        ax.add_artist(connection)

    plt.show()

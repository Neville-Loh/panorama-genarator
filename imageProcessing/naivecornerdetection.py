import math
from matplotlib import pyplot as plt
import numpy as np


def naiveDetection(image, threshold=100, plot_image=False):

    pixel_array = np.array(image)

    image_height, image_width = np.shape(pixel_array)

    corners = []
    right_pixel_shifts = [(1, 1), (0, 1), (-1, 1)]
    down_pixel_shifts = [(-1, -1), (-1, 0), (-1, 1)]

    for y in range(2, image_height - 2):
        for x in range(2, image_width - 2):
            delta_horizontal = 0
            for shift in right_pixel_shifts:
                delta_horizontal += (pixel_array[y, x] - pixel_array[y + shift[1], x + shift[0]]) // 3
            delta_vertical = 0
            for shift in down_pixel_shifts:
                delta_horizontal += (pixel_array[y, x] - pixel_array[y + shift[1], x + shift[0]]) // 3

            delta = math.sqrt(delta_horizontal ** 2 + delta_vertical ** 2)

            if delta > threshold:
                corners.append((x, y, delta))

    pq_n_best_corner_coor = [(corner[0], corner[1]) for corner in corners]

    if plot_image:
        plt.figure(figsize=(20, 18))
        plt.gray()
        plt.imshow(image)
        plt.scatter(*zip(*pq_n_best_corner_coor), s=1, color='r')
        plt.axis('off')
        plt.show()

    return corners

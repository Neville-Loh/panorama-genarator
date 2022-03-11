#!/usr/bin/env python

# pixelops.py - Image processing based on pixel arrays involving single pixel operations
#
# Copyright (C) 2020 Martin Urschler <martin.urschler@auckland.ac.nz>
#
# Original concept by Martin Urschler.
#
# LICENCE (MIT)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import imageProcessing.utilities as IPUtils


def scaleAndQuantize(pixel_array, image_width, image_height, min_value, max_value):

    output_pixel_array = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    scale_factor = 255.0 / (max_value - min_value)

    if max_value > min_value:
        for y in range(image_height):
            for x in range(image_width):
                value = int(round((pixel_array[y][x] - min_value) * scale_factor))
                if value < 0:
                    output_pixel_array[y][x] = 0
                elif value > 255:
                    output_pixel_array[y][x] = 255
                else:
                    output_pixel_array[y][x] = value

    return output_pixel_array



def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):

    (min_value, max_value) = IPUtils.computeMinAndMaxValues(pixel_array, image_width, image_height)

    print("before scaling, min value = {}, max value = {}".format(min_value, max_value))

    return scaleAndQuantize(pixel_array, image_width, image_height, min_value, max_value)



def scaleTo0And1(pixel_array, image_width, image_height):

    (min_value, max_value) = IPUtils.computeMinAndMaxValues(pixel_array, image_width, image_height)

    print("before scaling, min value = {}, max value = {}".format(min_value, max_value))

    output_pixel_array = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    scale_factor = 1.0 / (max_value - min_value)

    if max_value > min_value:
        for y in range(image_height):
            for x in range(image_width):
                output_pixel_array[y][x] = (pixel_array[y][x] - min_value) * scale_factor

    return output_pixel_array




#!/usr/bin/env python

# utilities.py - Utility functions for image processing based on pixel arrays
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

import sys

# r,g,b expected to be between 0 and 255 respectively.
# greyvalue will be an int between 0 and 255 as well.
def rgbToGreyscale(r, g, b):
    greyvalue = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    return greyvalue


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = []
    for row in range(image_height):
        new_row = []
        for col in range(image_width):
            new_row.append(initValue)
        new_array.append(new_row)

    return new_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = sys.maxsize
    max_value = -min_value

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] < min_value:
                min_value = pixel_array[y][x]
            if pixel_array[y][x] > max_value:
                max_value = pixel_array[y][x]

    return(min_value, max_value)


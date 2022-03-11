#!/usr/bin/env python

# convolve2D.py - Convolution operations on pixel arrays in 2D
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

def computeSeparableConvolution2DOddNTapBorderZero(pixel_array, image_width, image_height, kernelAlongX, kernelAlongY = []):

    if len(kernelAlongY) == 0:
        kernelAlongY = kernelAlongX

    intermediate = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    final = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    # two pass algorithm for separable convolutions

    kernel_offset = len(kernelAlongX) // 2
    #print("ntap kernel offset", kernel_offset)

    for y in range(image_height):
        for x in range(image_width):
            if x >= kernel_offset and x < image_width - kernel_offset:
                convolution = 0.0
                for xx in range(-kernel_offset, kernel_offset+1):
                    convolution = convolution + kernelAlongX[kernel_offset+xx] * pixel_array[y][x+xx]
                intermediate[y][x] = convolution

    kernel_offset = len(kernelAlongY) // 2

    for y in range(image_height):
        for x in range(image_width):
            if y >= kernel_offset and y < image_height - kernel_offset:
                convolution = 0.0
                for yy in range(-kernel_offset, kernel_offset+1):
                    convolution = convolution + kernelAlongY[kernel_offset+yy] * intermediate[y+yy][x]
                final[y][x] = convolution

    return final
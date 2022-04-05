# The Berg-Loh implementation of Histogram of Gradients Feature Descriptors
# This is based on work by Dalal & Triggs, 2005, originally presented in
# Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for
# human detection. In 2005 IEEE computer society conference on computer vision
# and pattern recognition (CVPR'05) (Vol. 1, pp. 886-893). Ieee.


# Pseudocode for the HOG Detector:
#      1. Gather an input image
#      2. Normalize gamma & colour
#      3. Compute Gradients
#      4. Weighted vote into spaital & orientation cells
#      5. Contrast normalize over overlapping spatial blocks
#      6. Collect HOG's over detection window
#      7. Linear SVM

# Notes:
#    While in their original work Dalal & Triggs used colour, for performance and simplicity reasons we will work in greyscale.


import numpy as np

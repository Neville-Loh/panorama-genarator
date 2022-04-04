import math
from typing import List, Type, Optional

from image_stiching.corner import Corner

"""
Data structure to store 2 Corner objects.
@Author: Neville Loh
"""


class Pair:
    """
    Pair of Corner objects.
    """
    corner1: Type[Corner]
    corner2: Type[Corner]
    ncc: float
    gradient: float
    distance: float

    def __init__(self, corner1: Type[Corner], corner2: Type[Corner], ncc: float):
        """Class Constructor
        Parameters
        ----------
        corner1: Corner
            First corner object.
        corner2: Corner
            Second corner object.
        ncc: float
            Normalized cross correlation value.
        """
        self.corner1 = corner1
        self.corner2 = corner2
        self.ncc: float = ncc
        self.gradient: float = self.cal_gradient()
        self.distance: float = self.cal_distance()

    def __lt__(self, other):
        return self.ncc < other.ncc

    def __eq__(self, other):
        return self.ncc == other.ncc

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"(({self.corner1.x}{self.corner1.y}) - ({self.corner2.x}{self.corner2.y}), {self.gradient})"

    def cal_gradient(self, width_offset: Optional[int] = 0) -> float:
        """
        Calculate the gradient of the pair.
        Parameters
        ----------
        width_offset: int
            Offset of the gradient.
        """
        return (self.corner1.y - self.corner2.y) / (self.corner1.x - self.corner2.x + width_offset)

    def cal_distance(self) -> float:
        """
        Calculate the distance of the pair.
        """
        return math.sqrt(
            (self.corner1.x - self.corner2.x) ** 2 + (self.corner1.y - self.corner2.y) ** 2)

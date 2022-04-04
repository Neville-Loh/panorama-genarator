import heapq
from typing import Tuple, List, Type
import numpy as np

"""
Class Corner

Represent a corner in the image. The key attribute include x,y coordinate
and the Harris Response which contains the intensity of how the pixel represent
a corner 

! NOTE, the natural order is reversed, where higher response will result in a lower natural order.
When applied sorting with this object, the result will be reversed. That is the sorted iteraterble will contain
the highest response corner. 

@Author: Neville Loh
"""

# Default image type is np array.
ImageArray = np.ndarray


class Corner:
    """Class Corner
        Represent a corner in the image
    """
    x: int = None
    y: int = None
    feature_descriptor: np.ndarray = None
    corner_response: float = 0.0
    patch_mse: float = 0.0

    def __init__(self, index: Tuple[int, int], corner_response: float):
        """Class Constructor
        Parameters
        ----------
        index :  Tuple[int, int]
            x and y coordinate of the pixel within the input image
        corner_response : float
            Harris Response which contains the intensity of how the pixel represent a corner
        """
        self.y, self.x = index
        self.corner_response = corner_response

    def __lt__(self, other):
        return self.corner_response > other.corner_response

    def __eq__(self, other):
        return self.corner_response == other.corner_response

    def __str__(self) -> str:
        return str((self.x, self.y, self.corner_response))

    def __repr__(self):
        return str(self)


def get_all_corner(img: ImageArray) -> List[Type[Corner]]:
    """
    Get all corner in the image
    Parameters
    ----------
    img : np.ndarray
        input image
    Returns
    -------
    List[Type[Corner]]
        List of all corner in the image
    """
    pq = []
    for index, val in np.ndenumerate(img):
        heapq.heappush(pq, Corner(index, val))
    return pq

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
    cornerness: float = 0.0

    def __init__(self, index: Tuple[int, int], cornerness: float):
        """Class Constructor
        Parameters
        ----------
        index :  Tuple[int, int]
            x and y coordinate of the pixel within the input image
        cornerness : float
            Harris Response which contains the intensity of how the pixel represent a corner
        """
        self.y, self.x = index
        self.cornerness = cornerness

    def __lt__(self, other):
        return self.cornerness > other.cornerness

    def __eq__(self, other):
        return self.cornerness == other.cornerness

    def __str__(self) -> str:
        return str((self.x, self.y, self.cornerness))

    def __repr__(self):
        return str(self)





def get_all_corner(img: ImageArray) -> List[Type[Corner]]:
    pq = []
    for index, val in np.ndenumerate(img):
        heapq.heappush(pq, Corner(index, val))
    return pq
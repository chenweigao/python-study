import numpy as np
from numpy.core._multiarray_umath import ndarray


def order_points(pts):
    # pts is a list of four points, specifying the (x, y) of each point in rect
    rect: ndarray = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    # The minimum value along a given axis.
    # top-left point, x+y sum, smallest
    rect[0] = pts[np.argmin(s)]

    # button-right point
    rect[2] = pts[np.argmax(s)]

    # x-y
    diff = np.diff(pts, axis=1)
    # top-right
    rect[1] = pts[np.argmin(diff)]
    # bottom-left
    rect[3] = pts[np.argmax(diff)]

    return rect

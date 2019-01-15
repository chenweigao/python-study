import cv2
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


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    # return the warped image
    return warped

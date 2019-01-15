import argparse

import cv2
import numpy as np

from transform import four_point_transform

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coord", help="comma separated list of source points")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args['coord']), dtype="float32")
warped = four_point_transform(image, pts)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

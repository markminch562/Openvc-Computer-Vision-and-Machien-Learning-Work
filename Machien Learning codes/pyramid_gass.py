# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:59:55 2019

@author: markm
"""

#creating image pyramids and sliding windows in python

#method 1 python and open cv for image pyramids

from skimage.transform import pyramid_gaussian
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to Image")
ap.add_argument("s", "--scale", type=float, default=1.5, help = "scale factor size")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])

     
for (i, resized ) in enumerate(pyramid_gaussian(image, downscale=2)):
     #if the image is too small break from loop
    if resized.shape[0] < 30 or resized.shape[1] < 30:
        break
     #show resized image
    cv2.imshow("layer {}".format(i+1), resized)
    cv2.waitKey(0)

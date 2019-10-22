# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 03:56:13 2019

@author: markm
"""

import cv2 as cv
import numpy as np


img_bgr = cv.imread('D:/opencvfold/pokeball.png')
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

template = cv.imread('D:/opencvfold/pokeball2.png', 0)
w,h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshhold = 0.4
loc =np.where(res >= threshhold)

for pt in zip(*loc[::-1]):
    cv.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0,255,255), 1)
cv.imshow('detected', img_bgr)
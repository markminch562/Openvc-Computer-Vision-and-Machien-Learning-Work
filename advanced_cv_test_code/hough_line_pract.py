# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 01:21:39 2019

@author: markm
"""

import cv2
import math
import numpy as np
import time
"use escape key to exsit images shown"
def main():"""
    lasttime = time.time()
    windowName = "preview"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False"""
    
src = cv2.imread('D:/opencvfold/road_test2.jpg', cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
if src is not None:
    ret = True
     
    while ret:
        
        
        dst = cv2.Canny(src, 50, 250, None, 3)
        
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        lines = cv2.HoughLines(dst, 1, np.pi/180, 350,None, 0, 0)
        if lines is not None:
            for i in range(0,len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a*rho
                y0 = b*rho
                pts1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pts2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pts1, pts2, (0,255,0), 3, cv2.LINE_AA)
        
        
    
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 10)
    
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
        
        
        cv2.imshow("picture", src)
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

        if cv2.waitKey(1) ==27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
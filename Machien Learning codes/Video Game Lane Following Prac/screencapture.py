# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 04:08:59 2019

@author: markm
"""

import numpy as np
#from PIL import ImageGrab
import cv2
import time
from mss import mss
#from directkeys import PressKey, W, A, S, D

def drawLines(img, lines):
    try:
        for line in lines:
            coords =  line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked



def process_img(c_img):
    #canny automaticly turns an image into gray scale so there is no need to do so this time
    processed_img = cv2.Canny(c_img, 50, 350, apertureSize = 3,L2gradient = True)
    #processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800, 300],[800, 500]])
    processed_img = roi(processed_img, [vertices])
    #                       edges         lines, np.180, theta, thershold, minlinelength, maxLineGap
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 100, 20, 15)
    # function call to draw lines on 
    drawLines(processed_img, lines)
    return processed_img


def main():
  
    #set sct to the screen capture command 
    sct = mss()
    lasttime = time.time() # create a namespace to tell programe speed
    bbox = {'top': 0, 'left': 1100, 'width': 800, 'height': 600}#set bounding box of screen capture
    print('happy')
    while(True):
       screen_cpt = sct.grab(bbox) #capture screen
       np_screen = np.array(screen_cpt) #convert to np array
       #cv2.imshow('window', np_screen)
       new_img = process_img(np_screen)
       cv2.imshow('window1', new_img)
       #display time of frame capture 
       print('play back is {} fps' .format(1/(time.time()-lasttime)))
       lasttime = time.time()
       #if q is pressed close window cv2.waitKey & 0xFF just makes sure the same key is read if caps lock is on
       if cv2.waitKey(5) & 0xFF == ord('q'): 
           cv2.destroyAllWindows()
           break
       
if __name__ == "__main__":
    main()      
    
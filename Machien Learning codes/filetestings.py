# -*- coding: utf-8 -*-
"""
Created on Thu May 23 03:15:28 2019

@author: markm
"""

#working with file systems
import cv2
import os
import shutil

dir1 = 'D:'
dir2 = 'imagepaths'

#new_dir = os.path.join(dir1 + os.sep, dir2, 'fun_images')

new_dir= 'D:\imagepaths'
new_dir = os.path.join(new_dir, 'fun_images')
new_dir2 = os.path.join(new_dir, 'holo.jpg')
print(new_dir)
print(new_dir2)
shaped = len(sorted(os.listdir(new_dir)))
listed = os.listdir(new_dir)

image_paths = [os.path.join(new_dir, listed[i])
                  for i in range(0, shaped)]

for i in range(shaped):
    if os.path.exists(image_paths[i]):
      images= cv2.imread(image_paths[i])
      cv2.imshow('holo', images)
      cv2.waitKey()  
    
    
cv2.destroyWindow('holo')




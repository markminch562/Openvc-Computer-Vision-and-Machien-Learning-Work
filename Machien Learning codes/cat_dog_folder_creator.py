# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 01:32:51 2019

@author: markm
"""

import os
import shutil




def copy_files(prefix_str, range_start,  range_end, target_dir):
    image_paths = [os.path.join(work_dir, 'train', prefix_str + '.' + str(i) + '.jpg')
                  for i in range(range_start, range_end)]
    dest_dir = os.path.join(work_dir, 'data', target_dir, prefix_str)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        for image_path in image_paths:
            shutil.copy(image_path, dest_dir)
    

work_dir = 'D:\imagepaths'

#createfiles(0, 100, 'doggey', start_path, 'data')
copy_files('dog', 0, 8000, 'train')
copy_files('cat', 0, 8000, 'train')


# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:05:18 2019

@author: markm
"""

import tensorflow as tf


x = tf.placeholder(tf.float32, name='x') 
y = tf.placeholder(tf.float32, name='y') 
z = tf.add(x, y, name='sum') 

session = tf.Session()

 
summary_writer = tf.summary.FileWriter('C:/opencvfold', session.graph) 
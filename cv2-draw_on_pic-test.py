#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:35:25 2019

@author: yue
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
shape_targets = np.loadtxt('recognizers/targets_symm.txt')
left_eye = (shape_targets[36]+shape_targets[39])/2
right_eye = (shape_targets[42]+shape_targets[45])/2
nose = shape_targets[33]
left_mouth = shape_targets[48]
right_mouth = shape_targets[54]
shape_targets = np.stack((left_eye,right_eye,nose,left_mouth,right_mouth))
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# Create new blank 300x300 red image
width, height = 224, 224

red = (0, 255, 0)
image = create_blank(width, height, rgb_color=red)
counter = 0
for point in shape_targets:
    print(point)
    cv2.circle(image, (int(point[0]),int(point[1])), 5,(255,0,0))
    
    if counter==1:
        break
    counter+=3
    
plt.imshow(image)
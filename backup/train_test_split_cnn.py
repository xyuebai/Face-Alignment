#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:39:48 2019

@author: yue
"""

#!/usr/bin/env python

import cv2
import time
import numpy as np
import dlib
import os
import sys
import datetime
import xml.etree.ElementTree as ET
import get_image_size

from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

if __name__ == '__main__':

    folder = "data_cnn"

    ann_file = folder + "/facial_5_all.xml"
    train_file = folder + "/facial_5_train.csv"
    test_file = folder + "/facial_5_test.csv"

    test_size = 0.05

    tree = ET.parse(ann_file)
    root = tree.getroot()
    images = root.findall("images")[0]

    np.random.seed(123)

    with open(test_file, "w") as fp:
        for image in images.iter("image"):

            if np.random.rand() < test_size:
                is_test_image = True
            else:
                is_test_image = False
            
            filename = image.get("file")
            
            try:
                im_width, im_height = get_image_size.get_image_size(filename)
            except get_image_size.UnknownImageFormat:
                print("Can't read " + filename)
                continue

            for box in image.iter("box"):
                
                left = int(box.get("left"))
                width = int(box.get("width"))
                top = int(box.get("top"))
                height = int(box.get("height"))

                box.set("left", str(left))
                box.set("top", str(top))
                box.set("width", str(width))
                box.set("height", str(height))

                for part in box.iter("part"):
                    
                    if part.get("name") == "left_eye":
                        x_left_eye = int(float(part.attrib["x"]))
                        y_left_eye = int(float(part.attrib["y"]))
                        part.attrib["x"] = str(x_left_eye)
                        part.attrib["y"] = str(y_left_eye)
                    if part.get("name") == "right_eye":
                        x_right_eye = int(float(part.get("x")))
                        y_right_eye = int(float(part.get("y")))
                        part.attrib["x"] = str(x_right_eye)
                        part.attrib["y"] = str(y_right_eye)
                    if part.get("name") == "nose":
                        x_nose = int(float(part.get("x")))
                        y_nose = int(float(part.get("y")))
                        part.attrib["x"] = str(x_nose)
                        part.attrib["y"] = str(y_nose)
                    if part.get("name") == "left_mouth":
                        x_left_mouth = int(float(part.get("x")))
                        y_left_mouth = int(float(part.get("y")))
                        part.attrib["x"] = str(x_left_mouth)
                        part.attrib["y"] = str(y_left_mouth)
                    if part.get("name") == "right_mouth":
                        x_right_mouth = int(float(part.get("x")))
                        y_right_mouth = int(float(part.get("y")))
                        part.attrib["x"] = str(x_right_mouth)
                        part.attrib["y"] = str(y_right_mouth) 
                        
                if is_test_image:
                    fp.write("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"\
                             % (filename, left, top, width, height, \
                                x_left_eye, y_left_eye, \
                                x_right_eye, y_right_eye, \
                                x_nose, y_nose, \
                                x_left_mouth, y_left_mouth, \
                                x_right_mouth, y_right_mouth))
            if is_test_image:
                images.remove(image)

    with open(train_file, "w") as fp:
        for image in images.iter("image"):
            
            filename = image.get("file")
            
            try:
                im_width, im_height = get_image_size.get_image_size(filename)
            except get_image_size.UnknownImageFormat:
                print("Can't read " + filename)
                continue

            for box in image.iter("box"):
                
                left = int(box.get("left"))
                width = int(box.get("width"))
                top = int(box.get("top"))
                height = int(box.get("height"))

                box.set("left", str(left))
                box.set("top", str(top))
                box.set("width", str(width))
                box.set("height", str(height))

                for part in box.iter("part"):
                    
                    if part.get("name") == "left_eye":
                        x_left_eye = int(float(part.attrib["x"]))
                        y_left_eye = int(float(part.attrib["y"]))
                        part.attrib["x"] = str(x_left_eye)
                        part.attrib["y"] = str(y_left_eye)
                    if part.get("name") == "right_eye":
                        x_right_eye = int(float(part.get("x")))
                        y_right_eye = int(float(part.get("y")))
                        part.attrib["x"] = str(x_right_eye)
                        part.attrib["y"] = str(y_right_eye)
                    if part.get("name") == "nose":
                        x_nose = int(float(part.get("x")))
                        y_nose = int(float(part.get("y")))
                        part.attrib["x"] = str(x_nose)
                        part.attrib["y"] = str(y_nose)
                    if part.get("name") == "left_mouth":
                        x_left_mouth = int(float(part.get("x")))
                        y_left_mouth = int(float(part.get("y")))
                        part.attrib["x"] = str(x_left_mouth)
                        part.attrib["y"] = str(y_left_mouth)
                    if part.get("name") == "right_mouth":
                        x_right_mouth = int(float(part.get("x")))
                        y_right_mouth = int(float(part.get("y")))
                        part.attrib["x"] = str(x_right_mouth)
                        part.attrib["y"] = str(y_right_mouth) 

                fp.write("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"\
                             % (filename, left, top, width, height, \
                                x_left_eye, y_left_eye, \
                                x_right_eye, y_right_eye, \
                                x_nose, y_nose, \
                                x_left_mouth, y_left_mouth, \
                                x_right_mouth, y_right_mouth))
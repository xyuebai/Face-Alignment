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

    folder = "data"

    ann_file = folder + "/facial_5_all.xml"
    train_file = folder + "/facial_5_train.xml"
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
                        x1 = int(float(part.attrib["x"]))
                        y1 = int(float(part.attrib["y"]))
                        part.attrib["x"] = str(x1)
                        part.attrib["y"] = str(y1)
                    else:
                        x2 = int(float(part.get("x")))
                        y2 = int(float(part.get("y")))
                        part.attrib["x"] = str(x2)
                        part.attrib["y"] = str(y2)
                        
                if is_test_image:
                    fp.write("%s,%d,%d,%d,%d,%d,%d,%d,%d\n" % (filename, left, top, width, height, x1, y1, x2, y2))
            
            if is_test_image:
                images.remove(image)

    with open(train_file, "w") as fp:
        fp.write(prettify(root))

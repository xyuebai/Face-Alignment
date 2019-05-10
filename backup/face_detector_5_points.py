#!/usr/bin/env python

import cv2
import time
import numpy as np
import dlib
import os
import sys
import datetime
import xml.etree.ElementTree as ET

from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def iou(box1, box2):

    x1, y1, w1, h1, score1 = box1
    x2, y2, w2, h2, score2 = box2

    # determine the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if the intersection is empty:
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = w1 * h1
    bb2_area = w2 * h2

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou
    
def inside(point, bbox):
    """ return true if point is inside bbox = (x,y,w,h) """

    x0, y0  = point
    x,y,w,h = bbox

    if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
        return True
    else:
        return False

def pad_img_to_fit_bbox(img, x, y, w, h):

    img = cv2.copyMakeBorder(img, - min(0, y), max(y+h - img.shape[0], 0),
                            -min(0, x), max(x+w - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y -= min(0, y)
    x -= min(0, x)

    return img, x, y, w, h

if __name__ == '__main__':

    frozen_graph = 'detection/240x180_depth075_ssd_mobilenetv1/frozen_inference_graph.pb'
    text_graph = 'detection/240x180_depth075_ssd_mobilenetv1/graph.pbtxt'
    cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)

    # Image input size, must match the network
    width = 240
    height = 180

    ann_file = "../../hyperface/aflw_cleaned_5_landmarks.csv"
    img_path = "../../hyperface/aflw_origin_folder_correct"

    root = ET.Element('dataset')
    comment = ET.Comment('Date ' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M"))
    root.append(comment)

    images = ET.SubElement(root, 'images')

    with open(ann_file) as fp:
        lines = fp.read().split("\n")

    num_lines = len(lines)

    with open(ann_file) as fp:
        for i, line in enumerate(fp):
            
            image = None

            if i == 0:
                continue

            parts = line.strip().split(",")
#            filename = parts[2]
#            filename = filename[22:].replace("/", "_")
#            filename = img_path + os.sep + os.path.basename(filename)
#            filename = os.path.abspath(filename)
            filename = parts[15]

            if not os.path.isfile(filename):
                print("Warning: File not found: %s" % filename)
                continue

            x_left_eye  = int(float(parts[5]))
            y_left_eye  = int(float(parts[6]))
            x_right_eye = int(float(parts[7]))
            y_right_eye = int(float(parts[8]))
            x_nose = int(float(parts[9]))
            y_nose = int(float(parts[10]))
            x_left_mouth = int(float(parts[11]))
            y_left_mouth = int(float(parts[12]))
            x_right_mouth = int(float(parts[13]))
            y_right_mouth = int(float(parts[14]))

            left_eye  = (x_left_eye, y_left_eye)
            right_eye = (x_right_eye, y_right_eye)
            nose = (x_nose, y_nose)
            left_mouth = (x_left_mouth, y_left_mouth)
            right_mouth = (x_right_mouth, y_right_mouth)
            # extend to 5...

            print("%.1f %% done..." % (100.0 * i / num_lines))
            img = cv2.imread(filename)
        
            rows, cols = img.shape[0:2]
            blob = cv2.dnn.blobFromImage(img, size=(width, height), swapRB=True, crop=False)
            cvNet.setInput(blob)        
            cvOut = cvNet.forward()

            detections = [] # Store all detections here to spot overlapping bboxes.

            # Iterate over all found bounding boxes:
            for detidx, detection in enumerate(cvOut[0, 0, :, :]):
        
                score = float(detection[2])
                if score > 0.3:
                    
                    left = int(detection[3] * cols)
                    top = int(detection[4] * rows)
                    right = int(detection[5] * cols)
                    bottom = int(detection[6] * rows)

                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    if right > cols:
                        right = cols
                    if bottom > rows:
                        right = rows

                    boxwidth = right - left
                    boxheight = bottom - top
            
                    box = [left, top, boxwidth, boxheight]

                    skip = False
                    new_bbox = box + [score]

                    for old_bbox in detections:
                        if iou(old_bbox, new_bbox) > 0.3:
                            print("Warning: overlapping boxes in img %s: %s and %s" % (filename, str(new_bbox), str(old_bbox)))
                            skip = True
                            break

                    if skip:
                        continue

                    detections.append(new_bbox)

                    if inside(left_eye, box) and inside(right_eye, box) and \
                    inside(nose, box) and inside(left_mouth, box) and \
                    inside(right_mouth, box) : # consider if we need to add the other 5 points?

                        if not image:
                            # Check if we have already an element with this filename
                            files = images.findall(".//image[@file='%s']" % filename)
                            
                            if files:
                                image = files[0]
                            else:
                                image = ET.SubElement(images, 'image')
                                image.set("file", os.path.abspath(filename))

                        box = ET.SubElement(image, "box")
                        box.set("left", str(left))
                        box.set("top", str(top))
                        box.set("height", str(boxheight))
                        box.set("width", str(boxwidth))

                        part = ET.SubElement(box, "part")
                        part.set("name", "left_eye")
                        part.set("x", str(x_left_eye))
                        part.set("y", str(y_left_eye))

                        part = ET.SubElement(box, "part")
                        part.set("name", "right_eye")
                        part.set("x", str(x_right_eye))
                        part.set("y", str(y_right_eye))
                        
                        part = ET.SubElement(box, "part")
                        part.set("name", "nose")
                        part.set("x", str(x_nose))
                        part.set("y", str(y_nose))
                        
                        part = ET.SubElement(box, "part")
                        part.set("name", "left_mouth")
                        part.set("x", str(x_left_mouth))
                        part.set("y", str(y_left_mouth))
                        
                        part = ET.SubElement(box, "part")
                        part.set("name", "right_mouth")
                        part.set("x", str(x_right_mouth))
                        part.set("y", str(y_right_mouth))
                    
                        # extend to 5 points...

    with open("data/facelandmark_5_all.xml", "w") as fp:
        fp.write(prettify(root))

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
from skimage import io
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from landmark_detector_trainer import train

def crop_face(img, rect, margin=0.2):
    x1 = rect.left()
    x2 = rect.right()
    y1 = rect.top()
    y2 = rect.bottom()
    # size of face
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # add margin
    full_crop_x1 = x1 - int(w * margin)
    full_crop_y1 = y1 - int(h * margin)
    full_crop_x2 = x2 + int(w * margin)
    full_crop_y2 = y2 + int(h * margin)
    # size of face with margin
    new_size_w = full_crop_x2 - full_crop_x1 + 1
    new_size_h = full_crop_y2 - full_crop_y1 + 1

    # ensure that the region cropped from the original image with margin
    # doesn't go beyond the image size
    crop_x1 = max(full_crop_x1, 0)
    crop_y1 = max(full_crop_y1, 0)
    crop_x2 = min(full_crop_x2, img.shape[1] - 1)
    crop_y2 = min(full_crop_y2, img.shape[0] - 1)
    # size of the actual region being cropped from the original image
    crop_size_w = crop_x2 - crop_x1 + 1
    crop_size_h = crop_y2 - crop_y1 + 1

    # coordinates of region taken out of the original image in the new image
    new_location_x1 = crop_x1 - full_crop_x1;
    new_location_y1 = crop_y1 - full_crop_y1;
    new_location_x2 = crop_x1 - full_crop_x1 + crop_size_w - 1;
    new_location_y2 = crop_y1 - full_crop_y1 + crop_size_h - 1;

    print(img.shape)
    new_img = np.random.randint(256, size=(new_size_h, new_size_w, img.shape[2])).astype('uint8')
    # new_img = np.random.rand(new_size_h, new_size_w, img.shape[2])

    new_img[new_location_y1: new_location_y2 + 1, new_location_x1: new_location_x2 + 1, :] = \
        img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1, :]

    # if margin goes beyond the size of the image, repeat last row of pixels
    if new_location_y1 > 0:
        new_img[0:new_location_y1, :, :] = np.tile(new_img[new_location_y1, :, :], (new_location_y1, 1, 1))

    if new_location_y2 < new_size_h - 1:
        new_img[new_location_y2 + 1:new_size_h, :, :] = np.tile(new_img[new_location_y2:new_location_y2 + 1, :, :],
                                                                (new_size_h - new_location_y2 - 1, 1, 1))
    if new_location_x1 > 0:
        new_img[:, 0:new_location_x1, :] = np.tile(new_img[:, new_location_x1:new_location_x1 + 1, :],
                                                   (1, new_location_x1, 1))
    if new_location_x2 < new_size_w - 1:
        plt.imshow(new_img)
        new_img[:, new_location_x2 + 1:new_size_w, :] = np.tile(new_img[:, new_location_x2:new_location_x2 + 1, :],
                                                                (1, new_size_w - new_location_x2 - 1, 1))

    return new_img

def dlib_bound_box(bbox):
    x,y,w,h = bbox
    rect = dlib.rectangle(left = x,
                          top = y,
                          right = x + w,
                          bottom = y + h)
    return rect
    

def landmark_detector(shape_model, img, rect):    

    
    # APPLY shape model
    shape = shape_model(img, rect)
    
    # Extract the landmarks
    
    landmarks_pred = []
    
    for k in range(shape.num_parts):
        x = shape.part(k).x
        y = shape.part(k).y
        landmarks_pred.append([x, y])
       
    
    landmarks_pred = np.array(landmarks_pred)
   
    return landmarks_pred

def five_points_aligner(rect, shape_targets, landmarks_pred, img, filename):
    
    
    B = shape_targets
    A = np.hstack((np.array(landmarks_pred), np.ones((len(landmarks_pred), 1))))
                
    a = np.row_stack((np.array([-A[0][1], -A[0][0], 0, -1]), np.array([
                     A[0][0], -A[0][1], 1, 0])))
    b=np.row_stack((-B[0][1],B[0][0]))

    for i in range(A.shape[0]-1):
        i += 1
        a = np.row_stack((a, np.array([-A[i][1], -A[i][0], 0, -1])))
        a = np.row_stack((a, np.array([A[i][0], -A[i][1], 1, 0])))
        b = np.row_stack((b,np.array([[-B[i][1]], [B[i][0]]])))
         
    X, res, rank, s = np.linalg.lstsq(a, b)
    cos = (X[0][0]).real.astype(np.float32)
    sin = (X[1][0]).real.astype(np.float32)
    t_x = (X[2][0]).real.astype(np.float32)
    t_y = (X[3][0]).real.astype(np.float32)
    scale = np.sqrt(np.square(cos)+np.square(sin))
    
    H = np.array([[cos, -sin, t_x], [sin, cos, t_y]])
    s = np.linalg.eigvals(H[:, :-1])
    R = s.max() / s.min()
    
    #X, res, rank, s = np.linalg.lstsq(A, B)
    #s = np.linalg.eigvals(X.T[:, :-1])
    #R = s.max() / s.min()
    
    if R < 2.0:
        warped = cv2.warpAffine(img, H, (224,224))
        #M = cv2.getRotationMatrix2D((112,112), 270, 1.0)
        #warped = cv2.warpAffine(warped, M, (224,224))
    else:
        # Seems to distort too much, probably error in landmarks
        # Let's just crop.
        print("Crop file %s." % filename)
        crop = crop_face(img, rect)
        warped = cv2.resize(crop, (224,224)) 
    
    return warped
                    
    


if __name__ == "__main__":
    
    options = dlib.shape_predictor_training_options()
    options.nu = 0.3 # OPTIMAL : 0.3
    options.oversampling_amount = 10 # OPTIMAL : 10 after that stays same
    options.num_test_splits = 50 # OPTIMAL : 32, higher slower
    options.be_verbose = True
    options.num_trees_per_cascade_level = 500 # Optimal : 500
    options.tree_depth = 4 # OPTIMAL : 4
    options.lambda_param = 0.01 # OPTIMAL : 0.01 (0.1 is okay and faster)
    options.feature_pool_size = 600 # OPTIMAL : 400 (Not much difference)
    options.cascade_depth = 15 # OPTIMAL : 15  (Not much difference)
    
    # foler of models
    pathto_training_xml = "data/facial_5_train_local.xml"
    pathto_models = "landmark_5_model"
    pathto_aligned_image = "alinged_image_noshear"
    
    # load targets
    shape_targets = np.loadtxt('recognizers/targets_symm.txt')
    left_eye = (shape_targets[36]+shape_targets[39])/2
    right_eye = (shape_targets[42]+shape_targets[45])/2
    nose = shape_targets[30]
    left_mouth = shape_targets[48]
    right_mouth = shape_targets[54]
    #shape_targets = np.stack((left_mouth,right_mouth,nose,left_eye,right_eye))
    
    # NB: the order in dlib predictor is from left to right , from top to down
    shape_targets = np.stack((left_eye,left_mouth,nose,right_eye,right_mouth))
    
    # load test datafile
    test_file = "data/facial_5_test_local.csv"
    # load / train landmark detection model
    models = glob.glob(pathto_models + "/*.dat")

    if models:
        shape_model_file = max(models, key=os.path.getctime)
    else:
        shape_model_file = train(pathto_training_xml, options, pathto_models)

    shape_model = dlib.shape_predictor(shape_model_file)
    
    with open(test_file, "r") as fp:
        
        for line in fp:
            
            parts = line.split(",")
            filename = parts[0]
            x,y,w,h = [int(p) for p in parts[1:5]]
            # load bound box / run face detector 
            bbox = [x,y,w,h]
            rect = dlib_bound_box(bbox)
            # load current image
            img = cv2.imread(filename)[..., ::-1]
            
            # run landmark detector
            landmarks_pred = landmark_detector(shape_model, img, rect)
            
            # apply least square
            img_wraped = five_points_aligner(rect, shape_targets, \
                                              landmarks_pred, img, filename)
            
            # save image
            if not os.path.isdir(pathto_aligned_image):
                os.makedirs(pathto_aligned_image)
                
            out_file = pathto_aligned_image + os.sep + os.path.basename(filename)[:-4] \
                        + ".jpg"
            #print("Saving image to %s" % out_file)
            cv2.imwrite(out_file, img_wraped[..., ::-1])
            
    
    
    
    
        
   
                        
                        
        
                   
                    
                    

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014
#
#   In particular, we will train a face landmarking model based on a small
#   dataset and then evaluate it.  If you want to visualize the output of the
#   trained model on some images then you can run the
#   face_landmark_detection.py example program with predictor.dat as the input
#   model.
#
#   It should also be noted that this kind of model, while often used for face
#   landmarking, is quite general and can be used for a variety of shape
#   prediction tasks.  But here we demonstrate it only on a simple face
#   landmarking task.
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

import os
import sys
import glob
import cv2
import numpy as np
import copy
import time
import dlib
import random

def train(training_xml_path, options = dlib.shape_predictor_training_options(),
          output_folder = "."):
    
    print("\nTraining with shape_predictor_training_options:\n")
#    print(" oversampling_amount: {}".format(options.oversampling_amount) + \
#          "\n nu: {}".format(options.nu) + \
#          "\n num_test_splits: {}".format(options.num_test_splits) + \
#          "\n num_trees_per_cascade_level: {}".format(options.num_trees_per_cascade_level) + \
#          "\n tree_depth: {}".format(options.tree_depth) + \
#          "\n be_verbose: {}".format(options.be_verbose) + \
#          "\n lambda_param: {}".format(options.lambda_param) + \
#          "\n cascade_depth: {}".format(options.cascade_depth) + \
#          "\n feature_pool_region_padding: {}".format(options.feature_pool_region_padding) + \
#          "\n feature_pool_size: {}".format(options.feature_pool_size) + \
#          "\n random_seed: {}".format(options.random_seed))
    
    output_file = output_folder + os.sep + "predictor_" + time.strftime("%y-%j-%H%M-%S") + ".dat"
    print("Output filename: {}".format(output_file))
    print("Training XML path: {}".format(training_xml_path))
        
    dlib.train_shape_predictor(training_xml_path, output_file, options)
    print("Training ready. Next Testing.")

    return output_file

def test(shape_model, img, bbox, true_landmarks, filename):    

    x,y,w,h = bbox
    img = img.copy()
    cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=[255, 100, 100], thickness=1)
    
    # Transform to dlib rectangles object
    rect = dlib.rectangle(left = x,
                          top = y,
                          right = x + w,
                          bottom = y + h)

    # APPLY shape model
    shape = shape_model(img, rect)
    
    # Extract the landmarks
    
    landmarks = []
    
    for k in range(shape.num_parts):
        x = shape.part(k).x
        y = shape.part(k).y
        landmarks.append([x, y])
        
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    landmarks = np.array(landmarks)

#    print("True landmarks: ")
#    print(true_landmarks)
#
#    print("Predicted landmarks: ")
#    print(landmarks)
#    
#    print("Bounding box given: ")
#    x,y,w,h = bbox
#    print([x,y,x+w,y+h])
    
    # compute error metric

    MAE = np.mean(np.hypot(true_landmarks[:,0]-landmarks[:,0], 
                           true_landmarks[:,1]-landmarks[:,1]))        

    return (MAE, img)

def run():

    options = dlib.shape_predictor_training_options()
    options.nu = 0.99 # OPTIMAL : 0.01
    options.oversampling_amount = 10 # OPTIMAL : 10 after that stays same
    options.num_test_splits = 32 # OPTIMAL : 32
    options.be_verbose = True
    options.num_trees_per_cascade_level = 500 # Optimal : 500
    options.tree_depth = 4 # OPTIMAL : 4
    options.lambda_param = 0.01 # OPTIMAL : 0.01 (0.1 is okay and faster)
    options.feature_pool_size = 600 # OPTIMAL : 400 (Not much difference)
    options.cascade_depth = 10 # OPTIMAL : 10  (Not much difference)
        
    # Train shape model 
    
    training_xml_path = "IMAGES_AT_SCALE_800/eyes_train.xml"
    output_folder = "trained_models"
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        
    models = glob.glob(output_folder + "/*.dat")

    if models:
        shape_model_file = max(models, key=os.path.getctime)
    else:
        shape_model_file = train(training_xml_path, options, output_folder)

    shape_model = dlib.shape_predictor(shape_model_file)
    
    # Test the accuracy on separate test data.
    
    test_file = "IMAGES_AT_SCALE_800/eyes_test.csv"
    errors = []
    
    with open(test_file, "r") as fp:
        
        for line in fp:
            
            parts = line.split(",")
            filename = parts[0]
            x,y,w,h = [int(p) for p in parts[1:5]]
            bbox = [x,y,w,h]
            
            landmarks = [int(p) for p in parts[5:]]
            landmarks = np.reshape(landmarks, (2,2))
            
            img = cv2.imread(filename)[..., ::-1]

            # Resize to 800x600
            target_width = 800
            aspect_ratio = float(img.shape[1]) / img.shape[0]
            target_height = int(target_width / aspect_ratio)
            
            img = cv2.resize(img, (target_width, target_height))                
            error, img_out = test(shape_model, img, bbox, landmarks, filename)
            errors.append(error)
            
            file_path =  os.path.curdir + os.sep + 'test_results'
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
                
            out_file = file_path + os.sep + os.path.basename(filename)[:-4] \
                       + "_" + str(int(100*error)) + ".jpg"
            #print("Saving image to %s" % out_file)
            cv2.imwrite(out_file, img_out[..., ::-1])
            
        print("Mean error is %.2f (pixels)" % (np.mean(errors)))
            
if __name__ == "__main__":
    
    run()
    
    
    

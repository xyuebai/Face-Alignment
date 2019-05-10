#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:24:02 2019

@author: yue
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import csv
import cv2
import sys
from tqdm import tqdm
from keras.models import load_model
import time
def preprocess(img):
    # RGB -> BGR

    x = img[..., ::-1].astype(np.float32)

    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    return x

def load_images(paths, labels, dects, target_size):
    X = []
    y = []
    dirs = []
    for i in tqdm(range(len(paths))):
        name = paths[i]
        name = name.strip().split('/',3)[3]
        name = '../../' + name
        

        if not os.path.isfile(name):
            print('File {0} does not exist'.format(name))
            continue
        
        img = cv2.imread(name)
        
        
        left, top, width, height = dects[i]
       
        label = labels[i]
        
        r_w = target_size[0] / float(width)
        r_h = target_size[1] / float(height)
        
        label = label.astype(np.float)
        
        
        label[0:10:2] = (label[0:10:2] - float(left)) * r_w
        label[1:10:2] = (label[1:10:2] - float(top)) * r_h
        

        crop_img = img[int(top):int(top)+int(height), int(left):int(left)+int(width)]
         
        
        
        x = cv2.resize(crop_img, target_size)
        x = x[..., ::-1]  # BGR -> RGB (which "preprocess" will invert back)

        x = preprocess(x)
        
        y.append(label)
        X.append(x)
        dirs.append(name)
        sys.stdout.write("-")
        sys.stdout.flush()
    
    X = np.array(X)
    y = np.array(y)


    return X, y, dirs


# load data from csv file to numpy format
# load data from csv file to numpy format
def load_data(path):
    dirs = []
    labels = []
    dects = []
    with open(path, 'r') as infile:
        data = csv.reader(infile, delimiter=',')
        for row in data:
            name = row[0]
            dect = row[1:5]
            label = row[5:]
            dects.append(dect)
            dirs.append(name)
            labels.append(label)
    dirs =np.array(dirs)
    labels = np.array(labels)
    dects = np.array(dects)
    return dirs, dects, labels


def evaluate(O_Net, y_pred, y_test, dect, img, target_size):
    left, top, width, height = dect
#    crop_img = img[int(top):int(top)+int(height), int(left):int(left)+int(width)]
    
    test_img = img.copy()
#    h, w, _ = test_img.shape
    r_w = target_size[0] / float(width)
    r_h = target_size[1] / float(height)
    pred_landmarks = []
    true_landmarks = []
    for k in range(5): 
        tl = (int(y_test[2*k] / r_w + float(left)), int(y_test[2*k+1] / r_h + float(top))) 
        pl = (int(y_pred[2*k] / r_w + float(left)), int(y_pred[2*k+1] / r_h + float(top))) 
        true_landmarks.append(tl)
        pred_landmarks.append(pl)
        cv2.circle(test_img, (int(tl[0]), int(tl[1])), 2, (0, 255, 0), -1)
        cv2.circle(test_img, (int(pl[0]), int(pl[1])), 2, (33, 45, 221), -1)
        
    true_landmarks = np.array(true_landmarks)
    pred_landmarks = np.array(pred_landmarks)
    MAE = np.mean(np.hypot((pred_landmarks[:,0]-true_landmarks[:,0])/int(width), 
                           (pred_landmarks[:,1]-true_landmarks[:,1])/int(height)))        

    return (MAE, test_img)

def evaluate_origin(O_Net, y_pred, y_test, img, target_size):
    img = img.copy()
    height, width, channels = img.shape
    t_w, t_h = target_size
    landmarks = []
    for k in range(5):
        tl = (y_test[2*k] / t_w * width, y_test[2*k+1] / t_h * height) 
        x = y_pred[2*k] / t_w * width
        y = y_pred[2*k+1] / t_h * height
        landmarks.append([x, y])
        cv2.circle(img, (int(tl[0]), int(tl[1])), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(x), int(y)), 2, (33, 45, 221), -1)
    landmarks = np.array(landmarks)
    MAE = np.mean(np.hypot((y_pred[0:10:2]-y_test[0:10:2])/width, 
                           (y_pred[1:10:2]-y_test[1:10:2])/height))        

    return (MAE, img)

def run():
    prepared_data = "train_test_data_fix"
    output_folder = "trained_models_cnn_fix" 
    model_trained = output_folder + os.sep + 'model-056-0.716316-0.967865.h5'
    testing_path = "data_cnn/facial_5_test.csv"
    file_path = output_folder + os.sep +'evaluate_acc'
    # input_size
    target_width = 48
    target_height = 48
    target_size = (target_width, target_height)
    
    print('Model Loading...')
    
    O_net = load_model(model_trained)
    
    print('Model Loaded')    
    
    
    print('Data Loading...')
    
    # load data
#        
#    test_dirs, test_dects, test_labels = load_data(testing_path)
#    np.save(prepared_data+'/dects_test.npy', test_dects)
    print('Data Loaded')
    # Prepare data
#    X_test, y_test, dirs_test = load_images(test_dirs, test_labels, test_dects, target_size)
#    np.save(prepared_data+'/X_test.npy', X_test)
#    np.save(prepared_data+'/y_test.npy', y_test)
#    np.save(prepared_data+'/dirs_test.npy', dirs_test)
    
    X_test = np.load('train_test_data_fix/X_test.npy')
    y_test = np.load('train_test_data_fix/y_test.npy')
    dects_test = np.load('train_test_data_fix/dects_test.npy')
    dirs_test = np.load('train_test_data_fix/dirs_test.npy')
    
    print('Data Prepared')
    
    y_pred = O_net.predict(X_test)
    errors = []
    start_time = time.time() # start time of the loop 
    counter = len(y_pred)
    for i in range(len(y_pred)):
        img_path = dirs_test[i]
        img = cv2.imread(img_path)[..., ::-1]    
        
        error, img_out = evaluate(O_net, y_pred[i], y_test[i], dects_test[i], img, target_size)
        
        errors.append(error)
        if not os.path.isdir(file_path):
                os.makedirs(file_path)
        img_name = img_path.rsplit('/',1)[1]
        out_file = file_path + os.sep + img_name[:-4] + "_" + str(int(100*error)) + ".jpg"
        cv2.imwrite(out_file, img_out[..., ::-1])
    
   
        
    print("FPS: ", counter / (time.time() - start_time))
    print("Mean error is %.4f (pixels)" % (np.mean(errors)))
        

   
            
if __name__ == "__main__":
    
    run()
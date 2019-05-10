#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:24:13 2019

@author: yue
"""
import os

import sys
import glob
import cv2
import numpy as np
import time
import dlib
import random
import csv
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from keras.utils import plot_model


global backend, layers, models, keras_utils

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
        
        out_file = name.rsplit('/',2)[0] + os.sep + 'test' 

        if not os.path.isfile(name):
            print('File {0} does not exist'.format(name))
            continue
        
        img = cv2.imread(name)
        
        
        left, top, width, height = dects[i]
       
        label = labels[i]
        
        r_w = target_size[0] / float(width)
        r_h = target_size[1] / float(height)
        
        label = label.astype(np.float)
#        test_img = img.copy()
#        for k in range(5):
#            
#            tl = (label[2*k] , label[2*k+1]) 
#            cv2.circle(test_img, (int(tl[0]), int(tl[1])), 2, (0, 255, 0), -1)
#            
#        cv2.imwrite(out_file + os.sep + name.rsplit('/',1)[1], test_img)
        
        
        label[0:10:2] = (label[0:10:2] - float(left)) * r_w
        label[1:10:2] = (label[1:10:2] - float(top)) * r_h
        
        
        crop_img = img[int(top):int(top)+int(height), int(left):int(left)+int(width)]
         
        
        # save cropped image and draw landmarkds
#        test_img = crop_img.copy()
#        for k in range(5):
#            
#            tl = (label[2*k] / r_w, label[2*k+1] / r_h) 
#            cv2.circle(test_img, (int(tl[0]), int(tl[1])), 2, (0, 255, 0), -1)
#        
#        
#        if not os.path.isdir(out_file):
#                os.makedirs(out_file)
#        cv2.imwrite(out_file + os.sep + name.rsplit('/',1)[1], test_img)
        
        
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

    return X

# load data from csv file to numpy format
def load_data(path, target_size):
    dirs = []
    labels = []
    dects = []
    with open(path, 'r') as infile:
        data = csv.reader(infile, delimiter=',')
        for row in data:
            
            name = row[0]
            name = name.strip().split('/',3)[3]
            name = '../../' + name
            if not os.path.isfile(name):
                print('File {0} does not exist'.format(name))
                continue 
            
            dect = row[1:5]
            
            label = np.array(row[5:])
            left, top, width, height = dect
            
            label = label.astype(np.float)
        
            r_w = target_size[0] / float(width)
            r_h = target_size[1] / float(height)
        

            label[0:10:2] = (label[0:10:2] - float(left)) * r_w
            label[1:10:2] = (label[1:10:2] - float(top)) * r_h
            
            dects.append(dect)
            dirs.append(name)
            labels.append(label)
    
    dirs =np.array(dirs)
    labels = np.array(labels)
    dects = np.array(dects)
    return dirs, dects, labels

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


    MAE = np.mean(np.hypot(true_landmarks[:,0]-landmarks[:,0], 
                           true_landmarks[:,1]-landmarks[:,1]))        

    return (MAE, img)
def onet():
     # Determine proper input shape
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
    model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(10, activation = 'linear', kernel_initializer='normal'))
    model.summary()
    
    return model

def my_train(model, dirs, true_labels):
    return



def run():
    # Train shape model 
    
    training_path = "data_cnn/facial_5_train.csv"
    testing_path = "data_cnn/facial_5_test.csv"
    prepared_data = "train_test_data_fix"
    output_folder = "trained_models_cnn_fix"
    model_trained = output_folder + os.sep + 'landmark.h5'
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        
    # Create O-Net model
    O_net = onet()
    optimizer = Adam(lr=0.001)
    O_net.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mae'])
    
    # input_size
    target_width = 48
    target_height = 48
    target_size = (target_width, target_height)
    
    print('Data Loading...')
    
    if False:
#    if os.path.isdir(prepared_data):
        X_train = np.load('train_test_data_fix/X_train.npy')
        y_train = np.load('train_test_data_fix/y_train.npy')
        # dects_train = np.load('train_test_data_fix/dects_train.npy')
        # dirs_train = np.load('train_test_data_fix/dirs_train.npy')
        X_test = np.load('train_test_data_fix/X_test.npy')
        y_test = np.load('train_test_data_fix/y_test.npy')
        # dects_test = np.load('train_test_data_fix/dects_test.npy')
        # dirs_test = np.load('train_test_data_fix/dirs_test.npy')
        
    else:
        if not os.path.isdir(prepared_data):
            os.makedirs(prepared_data)
        
        # load data
        train_dirs, train_dects, train_labels = load_data(training_path, target_size)
        np.save(prepared_data+'/dirs_train.npy', train_dirs)
        np.save(prepared_data+'/y_train.npy', train_labels)
        test_dirs, test_dects, test_labels = load_data(testing_path, target_size)
        np.save(prepared_data+'/y_test.npy', test_labels)
        np.save(prepared_data+'/dirs_test.npy', test_dirs)
        print('Data Loaded')
        
        # Prepare data
        X_train = load_images(train_dirs, train_labels, train_dects, target_size)
        np.save(prepared_data+'/X_train.npy', X_train)
        X_test = load_images(test_dirs, test_labels, test_dects, target_size)
        np.save(prepared_data+'/X_test.npy', X_test)
        
    
    print('Data Prepared')
    
    
    # Start Training
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    mcp_save = ModelCheckpoint(output_folder + os.sep + 
                               'model-{epoch:03d}-{mean_absolute_error:03f}-{val_mean_absolute_error:03f}.h5' , 
                               save_best_only=True, 
                               monitor='val_loss', 
                               mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, min_lr=1e-5)
                                       
                                       
                                       
    print('Training Starts...')
    history = O_net.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_data=(X_test, y_test), 
              callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    print('Training Over...')
    
    # Save the Model
    O_net.save(model_trained)
    # summarize history for accuracy
    f = plt.figure()
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    f.savefig("mae.pdf", bbox_inches='tight')
    # summarize history for loss
    f = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    f.savefig("loss.pdf", bbox_inches='tight')
    # Test the accuracy on separate test data.
            
if __name__ == "__main__":
    
    run()
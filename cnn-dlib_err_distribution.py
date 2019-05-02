#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:20:10 2019

@author: yue
"""
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})

directory = '/home/yue/workplace/python/preprocessing_II/test_images_5_points/test_results_nu0.3_new'

error_degrees = []
for filename in os.listdir(directory):
    error_degree = filename[:-4].rsplit('_',1)[1]
    error_degrees.append(int(error_degree))

cnt = Counter()    
for error in error_degrees:
    cnt[error] += 1
    
print(cnt)
#plt.bar(cnt.keys(), cnt.values(),  color='g')

# test 2

directory = '/home/yue/baiy-narvi/AFLW_convert_landmark/shape_predictor/trained_models_cnn_fix/evaluate'

error_degrees = []
for filename in os.listdir(directory):
    error_degree = filename[:-4].rsplit('_',1)[1]
    error_degrees.append(int(error_degree))

cnt1 = Counter()    
for error in error_degrees:
    cnt1[error] += 1
    
print(cnt1)
t_dlib = [k  for  k in  cnt.keys()]
t_cnn = [k+0.3 for  k in  cnt1.keys()]
fig = plt.gcf()
ax = plt.subplot(111)
w = 0.3
p1=ax.bar(t_dlib, cnt.values(), width=w,align='center')
p2=ax.bar(t_cnn, cnt1.values(),width=w,align='center')

ax.set_ylabel('the number of pictures',  fontsize=16)
ax.set_xlabel('percentage of deviation compare to the ground truth label (%)',  fontsize=16)

plt.axis([-1, 30, 0, 250])
plt.xticks(np.arange(0, 31, step=1))
fig.set_size_inches(18.5, 10.5)
plt.legend((p1[0], p2[0]), ('DLIB', 'CNN'), prop={'size': 20})
plt.savefig('dlib_cnn_landmark.png')
plt.show()
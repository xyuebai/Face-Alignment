#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:27:46 2019

@author: yue
"""
localdir= '/home/yue/workplace/python/preprocessing_II/'

import xml.etree.ElementTree as ET
tree = ET.parse('data/facial_5_all.xml')
root = tree.getroot()


for child in root:
    print(child.tag, child.attrib)
    
    for sub_child in child:       
        sub_child.attrib['file']= localdir+sub_child.attrib['file'].split('/',4)[4]



tree.write('data/facial_5_train_local.xml')

# %% csv

import csv

with open('data/facial_5_test.csv', 'r') as in_csv,  open('data/facial_5_test_local.csv' , 'w') as out_csv:
    writer = csv.writer(out_csv, delimiter=',')
    csv_reader = csv.reader(in_csv, delimiter=',')
    for row in csv_reader:
        row[0] = localdir+row[0].split('/',4)[4]
        writer.writerow(row)
        
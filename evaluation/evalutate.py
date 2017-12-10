#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:32:00 2017

@author: cruncher
"""
import sys
sys.path.append('../util')
from utils_eval import pixelAccuracy, intersectionAndUnion
from scipy.misc import imread
import numpy as np

dropbox_path    = '/home/cruncher/Picpart/dropbox/'
data_path       = dropbox_path + 'data/'
images_root     = data_path + 'ADEChallengeData2016/images/'
anno_root       = data_path + 'ADEChallengeData2016/annotations/'
anno_path       = anno_root + 'validation/'
pred_path       = images_root + 'predictions/' + '20171207_174405/'
list_file       = data_path + 'ADEChallengeData2016/validation.min.txt'
num_classes = 150

# Get the list of images
with open(list_file, 'r') as file:
    files = [line.rstrip() for line in file]
    
# Evaluate Images
PAs = np.zeros((len(files), 3))
Ints = np.zeros((len(files), num_classes))
Unions = np.zeros_like(Ints)
for i, file in enumerate(files):
    # Load the prediction and lable
    pred = imread(pred_path+file+'.png')
    lbl = imread(anno_path+file+'.png')
    
    pa = pixelAccuracy(pred,lbl)
    iau = intersectionAndUnion(pred,lbl,num_classes)
    
    PAs[i,:] = pa
    Ints[i,:] = iau[0]
    Unions[i,:] = iau[1]
    
IOUs = Ints.sum(0)/sum(Unions+sys.float_info.epsilon,0)
IOU = IOUs.mean()
ACC = sum(PAs[:,1])/sum(PAs[:,2])
score = np.mean((IOU, ACC))

print("Final score: {:.2f} with mean IOU: {:.2f} and mean ACC: {:.2f}".format(score, IOU, ACC))
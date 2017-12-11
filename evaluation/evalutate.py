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
from multiprocessing import Pool

dropbox_path    = '/home/cruncher/Picpart/dropbox/'
snapshot_folder = 'snapshots/DilatedNet_Dec09_1900' #'pretrained_models/dilated_net' | 'snapshots/DilatedNet_Dec09_0430'
dataset_name    = 'ADEChallengeData2016'
list_filename   = 'validation.txt'
anno_folder     = 'annotations/validation'
pred_folder     = 'predictions/20171210_193855'
results_file    = 'results.txt'
num_jobs = 18

data_path       = dropbox_path + 'data/'
images_root     = data_path + 'ADEChallengeData2016/images/'
pred_path       = dropbox_path + snapshot_folder + '/' + pred_folder + '/'
anno_path       = data_path + dataset_name + '/' + anno_folder + '/'
list_file       = data_path + dataset_name + '/' + list_filename
num_classes = 150

def eval_file(file):
    # Load the prediction and lable
    pred = imread(pred_path+file+'.png')
    lbl = imread(anno_path+file+'.png')
    
    pa = pixelAccuracy(pred,lbl)
    iau = intersectionAndUnion(pred,lbl,num_classes)
    return (pa, iau[0], iau[1])
    
    
# Get the list of images
with open(list_file, 'r') as file:
    files = [line.rstrip() for line in file]
    
# Evaluate Images
p = Pool(num_jobs)
result = list(zip(*p.map(eval_file, files)))

PAs = np.array(result[0])
Ints = np.array(result[1])
Unions = np.array(result[2])
    
IOUs = Ints.sum(0)/sum(Unions+sys.float_info.epsilon,0)
IOU = IOUs.mean()
ACC = sum(PAs[:,1])/sum(PAs[:,2])
score = np.mean((IOU, ACC))

# Save results
msg = "Final score: {:.2f} with mean IOU: {:.2f} and mean ACC: {:.2f}".format(score, IOU, ACC)
with open(pred_path+results_file, 'w') as file:
    file.write(msg)
print(msg)
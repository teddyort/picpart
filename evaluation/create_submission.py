#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:34:40 2017

@author: cruncher
"""

import caffe
from scipy.misc import toimage, imresize
import cv2
import os
import  numpy as np
import datetime
import time
import sys

dt = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
dropbox_path    = '/home/cruncher/Picpart/dropbox/'
save_foldername = 'pretrained_dilatednet_test' # dt
model_folder    = 'pretrained_models' #'snapshots/DilatedNet_Dec_05/'
proto           = 'deploy_DilatedNet.prototxt'
model           = 'DilatedNet_iter_120000.caffemodel'
dataset_name    = 'ADEChallengeData2016'
list_filename   = 'testing.txt' #validation.txt | list.txt
images_folder   = 'testing' #'testing | validation'
output_layer    = 'fc_final_up' # 'fc_final_up' | out
print_after     = 10
GPU_DEVICE      = 1

data_path       = dropbox_path + 'data/'
model_path      = dropbox_path + model_folder + '/'
images_path     = data_path + dataset_name + '/images/' + images_folder + '/'
pred_path       = data_path + dataset_name + '/images/predictions/' + save_foldername + '/'
list_file       = data_path + dataset_name + '/' + list_filename


# Choose device
caffe.set_mode_gpu()
caffe.set_device(GPU_DEVICE)

# Add model folder to path
sys.path.append(model_path)

# Create the network
net = caffe.Net(model_path+proto, model_path+model, caffe.TEST)

# Load the data # TODO subtract the mean
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.array([109.5388, 118.6897, 124.6901]))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# Get the list of images
with open(list_file, 'r') as file:
    files = [line.rstrip() for line in file]

# Ensure the prediction folder exists
if not os.path.exists(pred_path):
    os.makedirs(pred_path)
    
# Process Images
start_time = time.time()
for j,file in enumerate(files):
    im = caffe.io.load_image(images_path + file + '.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    # Forward Pass
    result = net.forward()[output_layer].squeeze()
    pred = result.argmax(0)
    pred = cv2.resize(pred, im.shape[1::-1], fx=0,fy=0,interpolation=cv2.INTER_NEAREST)

    #Save
    toimage(pred, cmin=0, cmax=255).save(pred_path+file+'.png')
    
    if j%print_after == 0 and j>0:
        print("Saved prediction {} after {:2f} seconds".format(j, time.time() - start_time))
print("Total Time: %s seconds" % (time.time() - start_time))

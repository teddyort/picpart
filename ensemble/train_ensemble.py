#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:00:37 2017

@author: cruncher
"""
import caffe
from scipy.misc import imread
import numpy as np
import sys
sys.path.append('../util')
from utils_eval import pixelAccuracy, intersectionAndUnion
from multiprocessing import Pool
import datetime, time
import cv2

dt = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
dropbox_path    = '/home/cruncher/Picpart/dropbox/'
save_foldername = dt
dataset_name    = 'ADEChallengeData2016'
model_folders   = ['pretrained_models/dilated_net', 'pretrained_models/fcn']
models          = ['DilatedNet_iter_120000.caffemodel', 'FCN_iter_160000.caffemodel']
protos          = ['deploy_DilatedNet.prototxt', 'deploy_FCN.prototxt']
output_layers   = ['fc_final_up', 'score']
list_filename   = 'validation.min.txt'
anno_folder     = 'annotations/validation'
images_folder   = 'validation' #'testing | validation'
pred_folder     = 'images/predictions'
num_classes     = 150
num_jobs        = 18
print_after     = 10
GPU_DEVICE      = 0
batch_size      = 1
params          = [0.5,0.5]

data_path       = dropbox_path + 'data/'
images_path     = data_path + dataset_name + '/images/' + images_folder + '/'
list_file       = data_path + dataset_name + '/' + list_filename
anno_path       = data_path + dataset_name + '/' + anno_folder + '/'
pred_path       = data_path + dataset_name + '/' + pred_folder + '/'

def evaluate(pred, lbl):
    pred = cv2.resize(pred, lbl.shape[1::-1], fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
    pa = pixelAccuracy(pred,lbl)
    iau = intersectionAndUnion(pred,lbl,num_classes)
    return (pa, iau[0], iau[1])

def evaluate_all(preds, labels):
    # Evaluate Images
    result = []
    for j, lbl in enumerate(labels):
        result.append(evaluate(preds[j], lbl))
    result = list(zip(*result))
    PAs = np.array(result[0])
    Ints = np.array(result[1])
    Unions = np.array(result[2])
    IOUs = Ints.sum(0)/sum(Unions+sys.float_info.epsilon,0)
    IOU = IOUs.mean()
    ACC = sum(PAs[:,1])/sum(PAs[:,2])
    return (IOU, ACC)
#    return np.mean((IOU, ACC))

# Choose device
caffe.set_mode_gpu()
caffe.set_device(GPU_DEVICE)

# Create the network and setup the preprocessor
nets, transformers = [],[]
for j in range(len(models)):
    proto_path = dropbox_path+model_folders[j]+'/'+protos[j]
    model_path = dropbox_path+model_folders[j]+'/'+models[j]
    net = caffe.Net(proto_path, model_path, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([109.5388, 118.6897, 124.6901]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    nets.append(net)
    transformers.append(transformer)

# Get the list of images
with open(list_file, 'r') as file:
    files = [line.rstrip() for line in file]
    
# Process a batch of images
start_time = time.time()
preds, labels = [],[]
for k in range(int(len(files)/batch_size)):
    # Read in the images and labels
    b_images, b_labels = [], []
    for j in range(k*batch_size, (k+1)*batch_size):
        b_images.append(caffe.io.load_image(images_path + files[j] + '.jpg'))
        b_labels.append(imread(anno_path + files[j] + '.png'))
        if j%print_after == 0 and j>0:
            print("Completed image {} after {:2f} seconds".format(j, time.time() - start_time))
            
    # Forward pass to probabilities
    probs = []
    for j, net in enumerate(nets):
        data = np.stack([transformers[j].preprocess('data', im) for im in b_images])
        result = net.forward_all(data=data)[output_layers[j]]
        probs.append(result)
    probs = np.stack(probs, -1)
    
    # Assemble the ensemble
    b_preds = np.tensordot(probs, params,1).argmax(1)
    b_preds = [x.squeeze() for x in np.vsplit(b_preds,b_preds.shape[0])]
    
    # Save for evaluation
    preds.extend(b_preds)
    labels.extend(b_labels)

# Evaluate all the images
(IOU, ACC) = evaluate_all(preds, labels)
score = np.mean([IOU,ACC])
print("Total Time: %s seconds" % (time.time() - start_time))
print("Final score: {:.4f} with mean IOU: {:.4f} and mean ACC: {:.4f}".format(score, IOU, ACC))
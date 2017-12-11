#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:00:37 2017

@author: cruncher
"""
import caffe
import numpy as np
from multiprocessing import Pool
import datetime
from do_ensemble import Ensemble
from scipy.optimize import minimize, basinhopping, brute
import os

dt = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
dropbox_path    = '/home/cruncher/Picpart/dropbox/'
save_foldername = dt
dataset_name    = 'ADEChallengeData2016'
model_folders   = ['pretrained_models/dilated_net', 'pretrained_models/fcn']
models          = ['DilatedNet_iter_120000.caffemodel', 'FCN_iter_160000.caffemodel']
protos          = ['deploy_DilatedNet.prototxt', 'deploy_FCN.prototxt']
output_layers   = ['fc_final_up', 'score']
list_filename   = 'training.min.txt'
anno_folder     = 'annotations/training'
images_folder   = 'training' #'testing | validation'
out_folder     = 'ensemble_weights'
output_filename = 'ensemble_'+dt+'.txt'
num_classes     = 150
num_jobs        = 18
print_after     = 10
GPU_DEVICE      = 0
batch_size      = 1
p0              = 0.8*np.ones(num_classes+1)

data_path       = dropbox_path + 'data/'
images_path     = data_path + dataset_name + '/images/' + images_folder + '/'
list_file       = data_path + dataset_name + '/' + list_filename
anno_path       = data_path + dataset_name + '/' + anno_folder + '/'
output_path     = data_path + dataset_name + '/' + out_folder + '/'

class MyBounds(object):
    def __init__(self, xmax=[1]*len(p0), xmin=[0]*len(p0) ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin
mybounds = MyBounds()

X,F,A = [],[],[]
def mycallback(x, f, accept):
    X.append(x)
    F.append(f)
    A.append(accept)

# Ensure the output folder exists 
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
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
    
paths = (images_path, anno_path, output_path)
opts = (num_classes, batch_size, print_after)
ensemble = Ensemble(nets, transformers, output_layers, files, paths, opts)

res = minimize(ensemble.do_ensemble, p0, method='Powell', options={'maxfev': 5, 'xtol':0.1, 'ftol':0.1})

#res = basinhopping(ensemble.do_ensemble, p0, niter=20, T=0.1, stepsize=0.4, callback=mycallback, disp=True, accept_test=mybounds)

#res = brute(ensemble.do_ensemble, ((0,1),)*len(p0), 3, full_output=True, disp=True)

np.savetxt(output_path + output_filename, res.x)

#ensemble.do_ensemble(p0)

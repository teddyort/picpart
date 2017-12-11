#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 03:38:04 2017

@author: cruncher
"""
import cv2
import sys
sys.path.append('../util')
from utils_eval import pixelAccuracy, intersectionAndUnion
import numpy as np
import time
import caffe
from scipy.misc import imread

class Ensemble(object):
    def __init__(self, nets, transformers, output_layers, files, paths, opts):
        self.nets = nets
        self.transformers = transformers
        self.output_layers = output_layers
        self.files = files
        self.images_path, self.anno_path, self.log_path = paths
        self.num_classes, self.batch_size, self.print_after = opts
    
    def evaluate(self, pred, lbl):
        pred = cv2.resize(pred, lbl.shape[1::-1], fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
        pa = pixelAccuracy(pred,lbl)
        iau = intersectionAndUnion(pred,lbl,self.num_classes)
        return (pa, iau[0], iau[1])
    
    def evaluate_all(self, preds, labels):
        # Evaluate Images
        result = []
        for j, lbl in enumerate(labels):
            result.append(self.evaluate(preds[j], lbl))
        result = list(zip(*result))
        PAs = np.array(result[0])
        Ints = np.array(result[1])
        Unions = np.array(result[2])
        IOUs = Ints.sum(0)/sum(Unions+sys.float_info.epsilon,0)
        IOU = IOUs.mean()
        ACC = sum(PAs[:,1])/sum(PAs[:,2])
        return (IOU, ACC)
        
    def do_ensemble(self, params):
        params = np.clip(params, 0, 1)
        w = np.vstack((params, 1-params)).T.reshape((-1, 1, 1, 2))
        # Process a batch of images
        start_time = time.time()
        files = self.files
        preds, labels = [],[]
        for k in range(int(len(files)/self.batch_size)):
            # Read in the images and labels
            b_images, b_labels = [], []
            for j in range(k*self.batch_size, (k+1)*self.batch_size):
                b_images.append(caffe.io.load_image(self.images_path + files[j] + '.jpg'))
                b_labels.append(imread(self.anno_path + files[j] + '.png'))
                if j%self.print_after == 0 and j>0:
                    print("Completed image {} after {:2f} seconds".format(j, time.time() - start_time))
                    
            # Forward pass to probabilities
            probs = []
            for j, net in enumerate(self.nets):
                data = np.stack([self.transformers[j].preprocess('data', im) for im in b_images])
                result = net.forward_all(data=data)[self.output_layers[j]]
                probs.append(result)
            probs = np.stack(probs, -1)
            
            # Assemble the ensemble
            b_preds = (probs*w).sum(-1).argmax(1)
            b_preds = [x.squeeze() for x in np.vsplit(b_preds,b_preds.shape[0])]
            
            # Save for evaluation
            preds.extend(b_preds)
            labels.extend(b_labels)
        
        # Evaluate all the images
        (IOU, ACC) = self.evaluate_all(preds, labels)
        score = np.mean([IOU,ACC])
        
        # Save to log
        with open(self.log_path+'ensemble_log.txt', "ba") as logfile:
            np.savetxt(logfile, np.hstack((np.array([score]), w[...,0].squeeze())).reshape(1,-1), delimiter='\t')
        runtime = time.time() - start_time
        print("Params: {}, Final score: {:.4f}, IOU: {:.4f}, ACC: {:.4f}, Time: {:.2f}".format(w.mean(0).squeeze(), score, IOU, ACC, runtime))
        return(1-score)
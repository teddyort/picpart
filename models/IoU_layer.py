# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:06:08 2017

@author: amado
"""
import caffe
import numpy as np
import scipy.misc # for saving images to disk
#import sys
#sys.path.append('../util/')
#from utils_eval import intersectionAndUnion


np.seterr(divide='ignore', invalid='ignore')

class IoULayer(caffe.Layer):
    """
    Layer for reporting the Intersection over Union measure to the console
    """
    def setup(self, bottom, top):
        """
        Check correct number of bottom inputs
        """
        if len(bottom) != 2:
            raise Exception('Need two bottom inputs for IoULayer')
        
        if len(top) != 1:
            raise Exception('Need one top layer for meanIoU')
        
        params = eval(self.param_str)
        # number of pixel classes including 0 which is not counted
        self.n_classes= params["classes"]
        self.verbose = params["verbose"]
        self.idx = 1
    
    def reshape(self, bottom, top):
        # each entry (i,j) in hist counts the number of pixels classified as i
        #   in the label and predicted as j (a perfect prediction is diagonal)
        self.hist = np.zeros([self.n_classes, self.n_classes])
        top[0].reshape(1)
    
    def fast_hist(self, labels, predictions):
        # todo check dimensions and min and max values
        return np.bincount(self.n_classes*labels.astype(int) + predictions, minlength=self.n_classes**2).reshape(self.n_classes,self.n_classes)
    
    def forward(self, bottom, top):
        """
        Compute IoU for current batch
        """
        # predictions must go first in the prototxt definition
        predictions = bottom[0].data
        labels = bottom[1].data
        batch_size = labels.shape[0]
        for i in range(batch_size):
            hist = self.fast_hist(labels[i,...].flatten(), predictions[i,...].argmax(0).flatten())
            self.hist += hist
        IoU = np.diag(self.hist)[1:]/(self.hist.sum(1)[1:]+self.hist[1:,:].sum(0)[1:]-np.diag(self.hist)[1:])
        # compute the mean only among the labels that are present
        present = self.hist[1:,:].sum(1)>0
        meanIoU = np.mean(np.nan_to_num(IoU[present]))
        
        # compare this IoU function with the one provided (intersectionAndUnion)
#        area_intersection = np.zeros([150,batch_size])
#        area_union = np.zeros([150,batch_size])
#        for i in range(batch_size):
#            (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(predictions[i,...].argmax(0), labels[i,...],150)
#        IoU2 = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
#        meanIoU2 = np.mean(IoU2)
        
        top[0].data[...] = meanIoU
        
        p0 = None
        if self.verbose:
            p0 = predictions[0,...].argmax(0)
            print('First 20 predictions of first image in batch:')
            print(p0.flatten()[0:20])
            print('Top 10 x 4 of hist:')
            print(self.hist[0:10,0:4])
        if self.verbose or self.phase == 1:
            print('{} classes present'.format(np.sum(present)))
            print('mean IoU: {}'.format(meanIoU))
#            print('mean IoU2:{}'.format(meanIoU2))  
        
#        print("saving image {}".format(self.idx))
#        self.saveImage(p0,"prediction_{}.png".format(self.idx))
#        self.saveImage(labels[0,0,...],"label_{}.png".format(self.idx))
#        self.idx += 1
        
    def saveImage(self, img, file_name):
        # for debugging only
        img = img.astype('uint16')
        img = scipy.misc.toimage(img, high=np.max(img), low=np.min(img), mode='I')
        img.save(file_name)
    
    def backward(self, top, propagate_down, bottom):
        """
        This layer does not backpropagate
        """
        pass

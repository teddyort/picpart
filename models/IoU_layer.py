# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:06:08 2017

@author: amado
"""
import caffe
import numpy as np

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
        
        params = eval(self.param_str)
        # number of pixel classes including 0 which is not counted
        self.n_classes= params["classes"]
        # keep track of the total histogram of precision
        #   each entry (i,j) counts the number of pixels classified as i
        #   in the label and predicted as j (a perfect prediction is diagonal)
        self.total_hist = np.zeros([self.n_classes, self.n_classes])
    
    def reshape(self, bottom, top):
        pass
    
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
        hist = self.fast_hist(labels.flatten(), predictions.argmax(1).flatten())
        self.total_hist += hist
        IoU = np.diag(hist)[1:-1]/(hist.sum(1)[1:-1]+hist.sum(0)[1:-1]-np.diag(hist)[1:-1])
        print('mean IoU: {}'.format(np.nanmean(IoU)))
        
    
    def backward(self, top, propagate_down, bottom):
        """
        This layer does not backpropagate
        """
        pass
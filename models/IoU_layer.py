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
    
    def reshape(self, bottom, top):
        # each entry (i,j) in hist counts the number of pixels classified as i
        #   in the label and predicted as j (a perfect prediction is diagonal)
        self.hist = np.zeros([self.n_classes, self.n_classes])
    
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
        for i in range(labels.shape[0]):
            hist = self.fast_hist(labels[i,...].flatten(), predictions[i,...].argmax(0).flatten())
            self.hist += hist
        IoU = np.diag(hist)[1:]/(hist.sum(1)[1:]+hist.sum(0)[1:]-np.diag(hist)[1:])
        print('IoU:')
        print(IoU)
        print('mean IoU: {}'.format(np.nanmean(IoU)))
        
    
    def backward(self, top, propagate_down, bottom):
        """
        This layer does not backpropagate
        """
        pass

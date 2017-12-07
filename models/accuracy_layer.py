# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:05:26 2017

@author: amado
"""

import caffe
import numpy as np

class SegmentationAccuracy(caffe.Layer):
    """
    Layer for reporting the Segmentation accuracy
    """
    def setup(self, bottom, top):
        """
        Check correct number of bottom inputs
        """
        if len(bottom) != 2:
            raise Exception('Need two bottom inputs for SegmentationAccuracy')
        
        params = eval(self.param_str)
        self.ignore_label = params['ignore_label']
    
    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        """
        Compute IoU for current batch
        """
        # predictions must go first in the prototxt definition
        predictions = bottom[0].data
        labels = bottom[1].data
        hits = np.zeros(bottom[1].data.shape[1:])
        n = 0
        for i in range(labels.shape[0]):
            mask_ignore = (labels[i,...] == self.ignore_label)
            hits += ((labels[i,...] == predictions[i,...].argmax(0)) & np.logical_not(mask_ignore))
            n += np.logical_not(mask_ignore).sum()
        mean_accuracy = hits.sum()/n
        print('mean pixel accuracy: {:.3f}'.format(mean_accuracy))
        
    
    def backward(self, top, propagate_down, bottom):
        """
        This layer does not backpropagate
        """
        pass


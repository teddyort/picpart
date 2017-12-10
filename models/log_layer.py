# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:48:53 2017

@author: amado
"""
import datetime
import caffe
import numpy as np

class LogLayer(caffe.Layer):
    """
    Layer for logging measures such as loss, accurcay, etc. to a text file
    """
    def setup(self, bottom, top):
        """
        Check correct number of bottom inputs
        """
        if len(bottom) == 0:
            raise Exception('Need at least one bottom layer for logging')
        
        params = eval(self.param_str)
        self.file = params['file']
        headers = params['headers']
        
        print('Logging data to '+self.file)
        
        with open(self.file,'w') as writer:
        
            writer.write('iteration,datetime')
            for h in headers:
                writer.write(','+h)
            writer.write('\n')
            
        self.iter = 0
    
    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        """
        Compute IoU for current batch
        """
        with open(self.file,'a') as writer:
            writer.write('{}'.format(self.iter))
            self.iter += 1
            writer.write(datetime.datetime.now().strftime(",%Y-%m-%d_%H:%M:%S"))
            for i in range(len(bottom)):
                writer.write(',{}'.format(np.asscalar(bottom[i].data)))
            writer.write('\n')
            
        
    
    def backward(self, top, propagate_down, bottom):
        """
        This layer does not backpropagate
        """
        pass



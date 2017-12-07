# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:26:36 2017

@author: amado
"""
import h5py
import sys
import scipy.misc
import numpy as np

sys.path.append('../../')
from paths import getDataPath

#F.close()
def createH5(params):
    output_file = params['name']+'.h5'    
    F = h5py.File(output_file,"w")
    
    files = open(params['data_list'], 'r').read().splitlines()
    N = len(files)
    
    print('{} {} images found'.format(N,params['name']))
    
    F.create_dataset("images",(N,params['resize'],params['resize'],3),dtype='uint8')
    F.create_dataset("labels",(N,params['resize'],params['resize']),dtype='uint8')
     
    for i in range(N):
        image = scipy.misc.imread(params['im_folder']+files[i]+'.jpg')
        if image.ndim == 2:
            image = np.repeat(image[:,:,None],3,axis=2)
        if image.ndim != 3 or image.shape[2] != 3:
            F.close()
            raise Exception('Channel size error reading image {}'.format(files[i]))
        label = scipy.misc.imread(params['lb_folder']+files[i]+'.png')
        if len(label.shape) != 2:
            F.close()
            raise Exception('Channel size error reading label {}'.format(files[i]))
        F["images"][i] = scipy.misc.imresize(image,(params['resize'],params['resize']))
        F["labels"][i] = scipy.misc.imresize(label,(params['resize'],params['resize']))
        
        if i % 100 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    
    F.close()
    print('Created H5 dataset file: {}'.format(output_file))
    
if __name__ == '__main__':
    params_train = {
        'name': 'training',
        'resize': 384,
        'im_folder': getDataPath()+'ADEChallengeData2016/images/training/',
        'lb_folder': getDataPath()+'ADEChallengeData2016/annotations/training/',
        'data_list': getDataPath()+'ADEChallengeData2016/training.txt'
    }
    params_val = {
        'name': 'validation',
        'resize': 384,
        'im_folder': getDataPath()+'ADEChallengeData2016/images/validation/',
        'lb_folder': getDataPath()+'ADEChallengeData2016/annotations/validation/',
        'data_list': getDataPath()+'ADEChallengeData2016/validation.txt'
    }
    createH5(params_train)
#    createH5(params_val)

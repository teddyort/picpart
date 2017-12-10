# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 01:04:07 2017

@author: amado

Compute the weights for the 150 classes of pixels for the weighted loss layer

following http://arxiv.org/pdf/1511.00561v2.pdf
weight(c) = median_freq / freq(c) where freq(c) is the number of pixels of
class c divided by the total number of pixels in images where c is present, 
and median_freq is the median of these frequencies.
"""

import numpy as np
import sys
from PIL import Image

sys.path.append('../util/')
from paths import getDropboxPath

weights_file = getDropboxPath()+'data/ADEChallengeData2016/class_weights_python.txt'
nclass = 150
list_file = getDropboxPath()+'data/ADEChallengeData2016/training.txt'
label_folder = getDropboxPath()+'data/ADEChallengeData2016/annotations/training/'
label_files = [label_folder+line+'.png' for line in open(list_file,'r').read().splitlines()]
N = len(label_files)

class_pixels = np.zeros(nclass,dtype='int64')
total_pixels = np.zeros(nclass,dtype='int64')

for i in range(N):
    print('Calculating weights ... {} out of {}'.format(i, N))
    lab = np.array(Image.open(label_files[i]))
    numels = lab.size
    hist = np.int64(np.bincount(lab.flatten(),minlength=nclass+1)[1:])
    class_pixels += hist
    total_pixels += np.int64(numels*(hist>0))
    
freq = class_pixels.astype(np.float)/total_pixels
med = np.median(freq)
weights = med/freq

with open(weights_file,'w') as F:
    for i in range(nclass):
        F.write('    class_weighting: {:6.3f} \t# {:d}\n'.format(weights[i],i+1))



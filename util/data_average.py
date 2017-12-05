# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:27:12 2017

@author: amado
"""
import numpy as np
import os
import sys

sys.path.append('../../')
from paths import getDataPath

folder = getDataPath()+'ADEChallengeData2016/images/training/'

m = np.zeros([1,3])
files = os.listdir(folder)
N = len(files)

count = 1
for f in os.listdir(folder):
    im_path = folder+f
    im = cv2.imread(im_path)
    m += np.mean(im,axis=(0,1))
    print('calculating mean ... {} out of {}'.format(count, N))
    count += 1

print('data mean:')
print(m/count)

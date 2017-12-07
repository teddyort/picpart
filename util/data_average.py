# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:27:12 2017

@author: amado
"""
import numpy as np
import os
import sys
import cv2

sys.path.append('../../')
from paths import getDropboxPath

folder = getDropboxPath()+'data/ADEChallengeData2016/images/training/'

m = np.zeros([1,3])
files = os.listdir(folder)
N = len(files)

max_w = 0
min_w = 10000
max_h = 0
min_h = 10000

count = 1
for f in os.listdir(folder):
    im_path = folder+f
    im = cv2.imread(im_path)
    h, w, c = im.shape
    if w > max_w:
        max_w = w
    elif w < min_w:
        min_w = w
    if h > max_h:
        max_h = h
    elif h < min_h:
        min_h = h
    m += np.mean(im,axis=(0,1))
    print('calculating mean ... {} out of {}'.format(count, N))
    count += 1

print('data mean:')
print(m/count)

print('max width: {}'.format(max_w))
print('min width: {}'.format(min_w))
print('max height: {}'.format(max_h))
print('min height: {}'.format(min_h))
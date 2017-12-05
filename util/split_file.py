# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:45:34 2017

@author: amado
"""
import os
import sys

sys.path.append('../../')
from paths import getDataPath

split = 'training'
#split = 'validation'

#split_file = getDataPath()+split+".txt"
split_file = split+".txt"

folder = getDataPath()+'ADEChallengeData2016/images/'+split

files = os.listdir(folder)

file_object = open(split+'.txt','w')
for f in sorted(files):
    file_object.write(f[0:-4]+'\n')
    print(f[0:-4])
    
file_object.close()

print('Wrote split file to '+split_file)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:17:26 2017

@author: cruncher
"""

import matplotlib.pyplot as plt
from scipy.misc import imread
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()

# Check for the last used directory
last_dir = "~"
history_filename = os.path.expanduser('~') + '/.class_checker'
try:
    with open(history_filename, 'r') as history_file:
        last_dir = history_file.read()    
except:
    pass

filename = filedialog.askopenfilename(initialdir=last_dir)
    
print('Reading image: {}'.format(filename))

# Save the folder to the history
with open(history_filename, 'w') as history_file:
    history_file.write(os.path.dirname(filename))
    
im = imread(filename)
plt.imshow(im, interpolation='none')
plt.colorbar()
plt.show()

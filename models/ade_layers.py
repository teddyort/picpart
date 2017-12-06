import sys
import caffe
import numpy as np
from PIL import Image
import random

sys.path.append('../util/')
from paths import getDataPath

class AdeSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - ade_dir: path to ADE dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        

        """
        print("-----------------------------------------------------------")
        print("Phase: {}".format(self.phase))
        # config
        params = eval(self.param_str)
        self.data_path = getDataPath()
        self.ade_dir = self.data_path+'ADEChallengeData2016/' # add the path to the dataset
        self.split = params['split']
        self.split_dir = self.data_path+'ADEChallengeData2016/' # add path to the split files
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.fine_size = 96 # must be multiple of 8 for DilatedNet
        self.PHASE = params['phase']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = self.split_dir+'{}.txt'.format(self.split) 
        self.indices = open(split_f, 'r').read().splitlines()
        self.N = len(self.indices)
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, self.N-1)
            
        # this array contains the vertical and horizontal offset in the first
        # column and whether to generate new values on the third
        self.crop_sizes = np.ones([self.N,3],dtype=np.int)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_size, *self.data.shape)
        top[1].reshape(self.batch_size, *self.label.shape)


    def crop(self, raw_img):
        h = raw_img.shape[0]
        w = raw_img.shape[1]
        if self.crop_sizes[self.idx,2]:
            self.crop_sizes[self.idx,0] = random.randint(0,h-self.fine_size)
            self.crop_sizes[self.idx,1] = random.randint(0,w-self.fine_size)
            self.crop_sizes[self.idx,2] = 0
        else:
            self.crop_sizes[self.idx,2] = 1
        
        v_offset = self.crop_sizes[self.idx,0]
        h_offset = self.crop_sizes[self.idx,1]
        return raw_img[v_offset:v_offset+self.fine_size,h_offset:h_offset+self.fine_size,...]
        

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, self.N-1)
        else:
            self.idx += 1
            if self.idx == self.N:
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}images/{}/{}.jpg'.format(self.ade_dir, self.split, idx))
        
        in_ = self.crop(np.array(im, dtype=np.float32))
        if (in_.ndim == 2):
            in_ = np.repeat(in_[:,:,None], 3, axis = 2)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}annotations/{}/{}.png'.format(self.ade_dir, self.split, idx))
        label = self.crop(np.array(im, dtype=np.uint8))
        label = label[np.newaxis, ...]
        return label


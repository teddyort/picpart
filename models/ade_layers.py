import sys
import caffe
import numpy as np
from PIL import Image
import random

sys.path.append('../../')
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
        # config
        params = eval(self.param_str)
        self.data_path = getDataPath()
        self.ade_dir = self.data_path+'ADEChallengeData2016/' # add the path to the dataset
        self.split = params['split']
        self.split_dir = self.data_path+'ADEChallengeData2016/' # add path to the split files
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.fine_size = 96 # must be multiple of 8 for DilatedNet

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = self.split_dir+'{}.txt'.format(self.split) 
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
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
        im_path = '{}images/training/{}.jpg'.format(self.ade_dir, self.indices[self.idx])
        print(im_path)
        im = Image.open('{}images/training/{}.jpg'.format(self.ade_dir, self.indices[self.idx]))
        in_ = crop(np.array(im, dtype=np.float32))
        # DilatedNet expects size divisible by 8 so crop accordingly
        h = in_.shape[0]
        w = in_.shape[1]
        h = (h//8)*8
        w = (w//8)*8
        in_ = in_[0:h,0:w,:]        
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
        im = Image.open('{}annotations/training/{}.png'.format(self.ade_dir, self.indices[self.idx]))
        label = np.array(im, dtype=np.uint8)
        # DilatedNet expects size divisible by 8 so crop accordingly
        h = label.shape[0]
        w = label.shape[1]
        h = (h//8)*8
        w = (w//8)*8
        label = label[0:h,0:w]    
        label = label[np.newaxis, ...]
        return label

    def crop(self, raw_img):
        h = raw_img.shape[0]
        w = raw_img.shape[1]
        v_offset = random.randint(0,h-self.fine_size)
        h_offset = random.randint(0,w-self.fine_size)
        return raw_img[v_offset:v_offset+self.fine_size,h_offset:h_offset+self.fine_size,:]
        

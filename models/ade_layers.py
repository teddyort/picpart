import sys
import caffe
import numpy as np
import scipy
from PIL import Image
import h5py

sys.path.append('../util/')
from paths import getDropboxPath

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
        print("Phase: {}".format('TRAIN' if self.phase == 0 else 'TEST'))
        # config
        params = eval(self.param_str)
        self.data_path = getDropboxPath()+'data/ADEChallengeData2016/'
        self.ade_dir = self.data_path # add the path to the dataset
        self.split = params['split']
        self.split_dir = self.data_path # add path to the split files
        self.mean = np.array(params['mean'],dtype=np.float32)
        self.randomize = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.fine_size = params['fine_size'] # must be multiple of 8 for DilatedNet
        self.data_shape = (self.batch_size, 3, self.fine_size, self.fine_size)
        self.resize_mode = params['resize_mode']
        if not self.resize_mode in ['crop','scale']:
            raise Exception("ERROR: resize_mode ({}) not recognized.".format(self.resize_mode))
        self.label_shape = (self.batch_size, 1, self.fine_size, self.fine_size)        
        self.PHASE = params['phase']
        self.loader = params.get('loader','disk')

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.indices = None
        self.img_set = None
        self.lab_set = None
        self.N = 0
        if self.loader == 'disk':
            # load indices for images and labels
            split_f  = self.split_dir+'{}.txt'.format(self.split) 
            self.indices = np.array(open(split_f, 'r').read().splitlines(),np.object)
            self.N = self.indices.shape[0]
            print('Found {} images from {}'.format(self.N, split_f))
            print(self.randomize)
        elif self.loader == 'h5':
            h5file = '{}{}.h5'.format(self.data_path,self.split)
            F = h5py.File(h5file)
            self.img_set = np.array(F['images'])
            self.lab_set = np.array(F['labels'])
            self.N = self.img_set.shape[0]
            print('Loaded {} images from {}'.format(self.N, h5file))
        self.idx = 0

        if self.randomize:
            self.shuffle()
            
        # this array contains the vertical and horizontal offset in the first
        # column and whether to generate new values on the third
        self.crop_sizes = np.ones([self.N,3],dtype=np.int)
    
    def shuffle(self):
        np.random.seed(self.seed)
        perm = np.random.permutation(self.N)
        if self.loader == 'disk':
            self.indices[:,...] = self.indices[perm,...]
        elif self.loader == 'h5':
            self.img_set = self.img_set[perm]
            self.lab_set = self.lab_set[perm]

    def increment(self):
        # pick next input            
        self.idx += 1
        if self.idx == self.N:
            self.idx = 0
            if self.randomize:
                self.shuffle()
        

    def reshape(self, bottom, top):
        # load image + label image batch pair
        self.data = np.zeros(self.data_shape)
        self.label = np.zeros(self.label_shape)
        for i in range(self.batch_size):
            self.data[i,...] = self.load_image(self.idx)
            self.label[i,...] = self.load_label(self.idx)
            self.increment()
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data_shape)
        top[1].reshape(*self.label_shape)


    def crop(self, raw_img):
        h = raw_img.shape[0]
        w = raw_img.shape[1]
        if self.crop_sizes[self.idx,2]:
            self.crop_sizes[self.idx,0] = np.random.randint(0,h-self.fine_size)
            self.crop_sizes[self.idx,1] = np.random.randint(0,w-self.fine_size)
            self.crop_sizes[self.idx,2] = 0
        else:
            self.crop_sizes[self.idx,2] = 1
        
        v_offset = self.crop_sizes[self.idx,0]
        h_offset = self.crop_sizes[self.idx,1]
        return np.array(raw_img[v_offset:v_offset+self.fine_size,h_offset:h_offset+self.fine_size,...],dtype=np.float32)
        
    def scale(self, raw_img):
        return scipy.misc.imresize(raw_img,(self.fine_size,self.fine_size),interp='nearest')
    
    def resize(self, raw_img):
        if self.resize_mode == 'crop':
            return self.crop(raw_img)
        elif self.resize_mode == 'scale':
            return self.scale(raw_img)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label


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
        in_ = None
        if self.loader == 'disk':
            im = Image.open('{}images/{}/{}.jpg'.format(self.ade_dir, self.split, self.indices[idx]))
            in_ = self.resize(np.array(im)).astype(np.float32)
        elif self.loader == 'h5':
            in_ = self.resize(self.img_set[idx]).astype(np.float32)
            
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
        label = None
        if self.loader == 'disk':
            im = Image.open('{}annotations/{}/{}.png'.format(self.ade_dir, self.split, self.indices[idx]))
            label = self.resize(im)[np.newaxis, ...]
        elif self.loader == 'h5':
            label = self.resize(self.lab_set[idx])
        return label


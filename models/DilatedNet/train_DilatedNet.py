import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import sys

sys.path.append('../')

mode = 'GPU'

if mode == 'GPU':
    caffe.set_mode_gpu()
    caffe.set_device(0)

#net = caffe.Net('/home/amado/caffe-rc5/tutorial/conv.prototxt', caffe.TEST)
#im = np.array(Image.open('/home/amado/caffe-rc5/examples/images/cat_gray.jpg'))
#im_input = im[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*im_input.shape)
#net.blobs['data'].data[...] = im_input
#
#net.forward()
#
#net.save('/home/amado/caffe-rc5/tutorial/mymodel.caffemodel')

solver = caffe.get_solver('solver_DilatedNet.prototxt')
#
#solver.net.forward()  # train net
#solver.step(1)

solver.solve()

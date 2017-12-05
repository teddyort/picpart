import caffe
import yaml


with open('../util/config.yaml', 'r') as f:
    config = yaml.load(f)
    
GPU = config['GPU']

if GPU:
    caffe.set_mode_gpu()
    caffe.set_device(0)


#solver = caffe.get_solver('DilatedNet/solver_DilatedNet.prototxt')
solver = caffe.get_solver('FCN/solver_FCN.prototxt')


solver.solve()

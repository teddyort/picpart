import caffe
import yaml

#restore_from = '../snapshots/DilatedNet_Dec_05/snapshot_iter_190000'
restore_from = ''

with open('../util/config.yaml', 'r') as f:
    config = yaml.load(f)
    
GPU = config['GPU']

if GPU:
    caffe.set_mode_gpu()
    caffe.set_device(0)

solver = None
solver = caffe.get_solver('DilatedNet/solver_DilatedNet.prototxt')
if len(restore_from) > 0:
    solver.restore(restore_from+'.solverstate')
    print('restored session from {}'.format(restore_from))
#solver = caffe.get_solver('FCN/solver_FCN.prototxt')


solver.solve()

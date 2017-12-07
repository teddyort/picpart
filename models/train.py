import caffe
import yaml

with open('../util/config.yaml', 'r') as f:
    config = yaml.load(f)
    
#restore_from = '../snapshots/DilatedNet_Dec_05/snapshot_iter_190000.solverstate'
restore_from = ''

copy_from = '***.caffemodel'

    
GPU = config['GPU']

if GPU:
    caffe.set_mode_gpu()
    caffe.set_device(0)

solver = None
solver = caffe.get_solver('DilatedNet/solver_DilatedNet.prototxt')
if len(restore_from) > 0:
    solver.restore(restore_from)
    print('restored session from {}'.format(restore_from))
elif len(copy_from) > 0:
    solver.net.copy_from(copy_from)
    print('copied weights from {}'.format(copy_from))


solver.solve()

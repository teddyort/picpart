import os
import yaml
#os.environ['GLOG_minloglevel'] = '2' 
import caffe


with open('../util/config.yaml', 'r') as f:
    config = yaml.load(f)

network = 'DilatedNet'
#network = 'FCN'

solver_prototxt = ''
restore_from = ''
copy_from = ''
if network == 'DilatedNet':
    solver_prototxt = 'DilatedNet/solver_DilatedNet.prototxt'
    #copy_from = config['dropbox']+'pretrained_models/DilatedNet_iter_120000.caffemodel'
    #restore_from = '../snapshots/DilatedNet_Dec_05/snapshot_iter_190000.solverstate'
elif network == 'FCN':
    solver_prototxt = 'FCN/solver_FCN.prototxt'
    copy_from = config['dropbox']+'pretrained_models/FCN_iter_160000.caffemodel'
    #restore_from = '../snapshots/snapshot_iter_4000.solverstate'


GPU = config['GPU']

if GPU:
    caffe.set_mode_gpu()
    caffe.set_device(0)

solver = caffe.get_solver(solver_prototxt)
if len(restore_from) > 0:
    solver.restore(restore_from)
    print('restored session from {}'.format(restore_from))
elif len(copy_from) > 0:
    solver.net.copy_from(copy_from)
    print('copied weights from {}'.format(copy_from))


solver.solve()

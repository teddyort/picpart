import os
# cruncher
data_path = '~/picpart/data/'

# amado ubuntu
#data_path = '/home/amado/Dropbox (MIT)/Local/Fall17/Computer Vision/Scene_Segmentation/data/'

def getDataPath():
    if not os.path.isdir(data_path):
        raise ValueError('Path not found: {}'.format(data_path))
    else:
        return data_path

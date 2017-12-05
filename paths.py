import os

#computer = 'cruncher'
computer = 'amado'

def getDataPath():
    
    if computer == 'cruncher':
        data_path = '/home/cruncher/picpart/data/'
        
    elif computer == 'amado':
        data_path = '/home/amado/dropbox/Local/Fall17/Computer_Vision/Scene_Segmentation/data/'
        
    else:
        raise ValueError('Computer not recognized: {}'.format(computer))
    
    if not os.path.isdir(data_path):
        raise ValueError('Path not found: {}'.format(data_path))
    else:
        return data_path

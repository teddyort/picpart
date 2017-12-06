import os
import yaml
import sys

package_directory = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(package_directory,'config.yaml')

def getDataPath():
    config = {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f)
    except IOError:
        print('ERROR: Unable to open config.yaml, did you forget to create it?')
        sys.exit()
        
    data_path = config['data_path']
    
    if not os.path.isdir(data_path):
        raise ValueError('Path not found: {}'.format(data_path))
    else:
        return data_path

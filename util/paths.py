import os
import yaml
import sys

package_directory = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(package_directory,'config.yaml')

def getDropboxPath():
    config = {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f)
    except IOError:
        print('ERROR: Unable to open config.yaml, did you forget to create it?')
        sys.exit()
        
    dropbox_path = config['dropbox']
    
    if not os.path.isdir(dropbox_path):
        raise ValueError('Path not found: {}'.format(dropbox_path))
    else:
        return dropbox_path

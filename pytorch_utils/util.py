import shutil
import pickle
import torch
import os

def save_nets(dir,nets,to_pickle = None,overwrite = True):
    '''
        Create folder and save nets and pickle

        :param dir: folder to save dataset
        :param nets: dictionary of pytorch modules to save
        :param to_pickle: any pickable object, us to save other checkpoint data
        :param overwrite: overwrite dir if exists
    '''
    if(overwrite):
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass
    os.makedirs(dir)

    for key,val in nets.items():
        torch.save(val.state_dict(),dir + '/' + key + '.net')

    if to_pickle is not None:
        pickle.dump(to_pickle, open(dir + '/TrainingLog.pkl', 'wb'))

def load_nets(dir,nets,on_gpu=True):
    '''
        Load pretrained weights saved with save_nets()

        :param dir: folder where nets were saved
        :param nets: dictionary of pytorch modules to load
        :returns: nets
    '''
    for key,val in nets.items():
        if on_gpu:
            state_dict = torch.load(dir + '/' + key + '.net')
        else:
            state_dict = torch.load(dir + '/' + key + '.net', map_location=torch.device('cpu'))
        val.load_state_dict(state_dict)

    return nets

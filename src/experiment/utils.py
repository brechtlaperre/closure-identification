import h5py
from pathlib import Path
import pandas as pd
import numpy as np

def load_experiment_from_file(filename, dataframe=False):
    '''Load experiment from file
    Input:
        filename: path to file to load 
    Output:
        x_train, x_valid, x_test, y_train, y_valid, y_test
        x_test and y_test are dictionaries!
    '''

    x_tests_dct = {}
    y_tests_dct = {}

    with h5py.File(filename, 'r') as f:

        x_train = pd.DataFrame()
        x_valid = pd.DataFrame()

        for feat in f['train']['input'].keys():
            x_train[feat] = np.array(f['train']['input'][feat])
        for feat in f['valid']['input'].keys():
            x_valid[feat] = np.array(f['valid']['input'][feat])

        outputs = []
        for status in ['train', 'valid']:
            out = pd.DataFrame()
            
            for key in f[status]['output'].keys():
                out[key] = np.array(f[status]['output'][key])
            
            outputs.append(out)

        for key in f['test']['input'].keys():
            x_tests_dct[key] = np.array(f['test']['input'][key]['x'])
            y_test = pd.DataFrame()
            
            for sk in f['test']['output'][key].keys():
                y_test[sk] = np.array(f['test']['output'][key][sk])

            y_tests_dct[key] = y_test
            
    y_train, y_valid = outputs

    if not dataframe:
        x_train = x_train.values
        x_valid = x_valid.values
            
    return x_train, x_valid, x_tests_dct, y_train, y_valid, y_tests_dct

def save_prediction_to_file(filename, data_dict):
    with h5py.File(filename, 'w') as f:
        g = f.create_group('prediction')
        for sims in data_dict.keys():
            sim_g = g.create_group('{}'.format(sims))
            for key in data_dict[sims].keys():
                s = sim_g.require_dataset('{}'.format(key), data_dict[sims][key].shape, data_dict[sims][key].dtype)
                s[...] = data_dict[sims][key]

def save_training_error_to_file(filename, data_dict):
    with h5py.File(filename, 'w') as f:
        for key in data_dict.keys():
            g = f.create_group('{}'.format(key))
            for subkey in data_dict[key].keys():
                s = g.require_dataset('{}'.format(subkey), data_dict[key][subkey].shape, data_dict[key][subkey].dtype)
                s[...] = data_dict[key][subkey]

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_prediction(prediction_file, truth_file):
    all_predictions = {}
    with h5py.File(prediction_file, 'r') as f:
        for sims in f['prediction'].keys():
            prediction = pd.DataFrame()
            g = f['prediction'][sims]
            for key in g.keys():
                prediction[key] = np.array(g[key])
            all_predictions[sims] = prediction

    _,_,_,_,_, truth = load_experiment_from_file(truth_file)

    return all_predictions, truth
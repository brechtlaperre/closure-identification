r'''
author: Brecht Laperre
date: 28/06/21

This script creates datasets for different experiments by 
creating the final training set and transforming input and output data
The transformers and final datasets are stored in data/experiment

INPUT TRANFORMATION:
1. MinMaxScaler to [0,1]

OUTPUT TRANFORMATION:
1. Box-cox transform for diagonal of pressure tensor
2. StandardScaler

'''
import copy
from pathlib import Path
import argparse 
import pickle
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import yaml

from sample_processed_data import create_experiment_from_stored_sample
from data_utils import DataSet

def get_boxcox_lambdas():
    return {'Ppar': -0.3, 'Pper1': -0.45, 'Pper2': -0.5}

def reverse_transform(data, transformer_file):

    outputtr = pickle.load(open(transformer_file, 'rb'))
    inv_data = data.copy()
    for col in data.columns:
        inv_data[[col]] = outputtr[col].inverse_transform(inv_data[[col]])
    #inv_data = pd.DataFrame(inv_data, columns=data.columns)

    bx_lambdas = get_boxcox_lambdas()
    inv_data = box_cox_inverse_transform(inv_data, bx_lambdas)

    return inv_data

def box_cox_transform(data, lambdas_dict):
    
    for feature, lamb in lambdas_dict.items():
        if feature in data.columns:
            data[feature] = boxcox(data[feature].values, lamb)

    return data

def box_cox_inverse_transform(data, lambdas_dict):
    
    for feature, lamb in lambdas_dict.items():
        if feature in data.columns:
            if lamb != 0:
                data = data.loc[data[feature] < (-1/lamb)*0.95].copy()
            data[feature] = inv_boxcox(data[feature].values, lamb)

    return data

def save_experiment_to_file(filename, train_set, valid_set, x_test_dict, y_test_dict):

    in_sets = [train_set.x, valid_set.x]
    out_sets = [train_set.y, valid_set.y]
    
    with h5py.File(filename, 'w') as f_r:
        tr_g = f_r.create_group('train')
        val_g = f_r.create_group('valid')

        for i, gr in enumerate([tr_g, val_g]):
            gr_in = gr.create_group('input')
            gr_out = gr.create_group('output')

            for feat in in_sets[i].columns:
                dtype = in_sets[i][feat].dtype
                s = gr_in.create_dataset('{}'.format(feat), in_sets[i][feat].shape, dtype=dtype)
                s[...] = in_sets[i][feat].to_numpy().astype(dtype)
            
            for target in out_sets[i].columns:
                dtype = out_sets[i][target].dtype
                s = gr_out.create_dataset('{}'.format(target),out_sets[i][target].shape, dtype=dtype)
                s[...] = out_sets[i][target].to_numpy().astype(dtype)

        test_g = f_r.create_group('test')
        gr_in = test_g.create_group('input')
        gr_out = test_g.create_group('output')

        for key in x_test_dict.keys():
            s_in = gr_in.create_group('{}'.format(key))
            s_inx = s_in.require_dataset('x', x_test_dict[key].shape, dtype=x_test_dict[key].dtype)
            s_inx[...] = x_test_dict[key]

            s_out = gr_out.create_group('{}'.format(key))
            for target in y_test_dict[key].columns:
                dtype = y_test_dict[key][target].dtype
                s = s_out.create_dataset('{}'.format(target), y_test_dict[key][target].shape, dtype=dtype)
                s[...] = y_test_dict[key][target].to_numpy().astype(dtype)        

def create_experiments(exp_name, exp_files, experiment_species, target, features, bx=True, outputscaler=None):

    all_files = ['high_res_2', 'high_res_hbg', 'high_res_bg0', 'high_res_bg3']
    
    target.sort()
    features.sort()

    path = 'data/experiment/{}'.format(exp_name)
    Path(path).mkdir(parents=True, exist_ok=True)

    for specie in experiment_species:
        datafiles = []
        for dfile in exp_files:
            datafiles.append('data/sampled/{}_{}_sampled.h5'.format(dfile, specie))
        train, valid, _, _, _ = create_experiment_from_stored_sample(target, features, datafiles)
        
        ## Create transformer
        input_scaler = StandardScaler()
        output_scaler = {}

        ## Transform data
        x_train = input_scaler.fit_transform(train.x)
        x_valid = input_scaler.transform(valid.x)

        if bx:
            bx_lambdas = get_boxcox_lambdas()
            bx_lambdas = {'Ppar': 0, 'Pper1': 0, 'Pper2': 0}
            y_train = box_cox_transform(train.y, bx_lambdas)
            y_valid = box_cox_transform(valid.y, bx_lambdas)
        else:
            y_train = train.y
            y_valid = valid.y

        for targ in target:
            if not outputscaler:
                outscaler = StandardScaler()
                print('Using standardscaler')
            else:
                print('Using custom scaler')
                outscaler = copy.copy(outputscaler)
            y_train[[targ]] = outscaler.fit_transform(y_train[[targ]])
            y_valid[[targ]] = outscaler.transform(y_valid[[targ]])
            output_scaler[targ] = copy.copy(outscaler)

        y_train = pd.DataFrame(y_train, columns=target)
        y_valid = pd.DataFrame(y_valid, columns=target)

        x_train = pd.DataFrame(x_train, columns=features)
        x_valid = pd.DataFrame(x_valid, columns=features)

        t_train = DataSet(x=x_train, y=y_train)
        t_valid = DataSet(x=x_valid, y=y_valid)

        x_testdata = {}
        y_testdata = {}
        for simulation in all_files:
            testfile = 'data/sampled/{}_{}_sampled.h5'.format(simulation, specie)
            _, _, test, _, _ = create_experiment_from_stored_sample(target, features, testfile)
            
            x_test = input_scaler.transform(test.x)
            if bx:
                print('Using box cox')
                y_test = box_cox_transform(test.y, bx_lambdas)
            else:
                y_test = test.y
            #y_test = pd.DataFrame(y_test, columns=target)
            for targ in target:
                y_test[[targ]] = output_scaler[targ].transform(y_test[[targ]])
            
            x_testdata[simulation] = x_test
            y_testdata[simulation] = y_test

        ## save to h5 file
        save_experiment_to_file('{}/{}.h5'.format(path, specie), t_train, t_valid, x_testdata, y_testdata)

        ## save transformer
        pickle.dump(input_scaler, open('{}/inputtr_{}.pkl'.format(path, specie), 'wb'))
        pickle.dump(output_scaler, open('{}/outputtr_{}.pkl'.format(path, specie), 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Raw data parser')
    parser.add_argument('--config_file',  '-f',
                        dest="configfile",
                        metavar='string',
                        help =  'path to an experiment config file',
                        default='')

    args = parser.parse_args()

    configfile = args.configfile

    if args.configfile == '':
        print('Please provide an experiment configfile.')
        exit(1)

    with open(configfile, 'r') as stream:
        config = yaml.safe_load(stream)

    create_experiments(exp_name=config['experiment_name'], 
                       exp_files=config['datafiles'],
                       experiment_species=config['species'], 
                       target=config['target'], 
                       features=config['features'], 
                       bx=config['boxcox'], 
                       outputscaler=config['outputscaler']
                       )


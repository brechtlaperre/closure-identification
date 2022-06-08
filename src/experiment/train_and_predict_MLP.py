import sys
import argparse

import pandas as pd
import yaml
import pickle
import torch
import numpy as np

sys.path.append('.')

from src.model.MLP import MLP
from utils import save_prediction_to_file, save_training_error_to_file, create_folder, load_experiment_from_file

def train_mlp(config, datafile, destination, epochs, save_results=True, load_models=False, train_targets=None):

    create_folder(destination)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training model on {}'.format(device))

    if train_targets is None:
        targets = config['hyperparameters'].keys()
    else:
        targets = train_targets
    x_train, x_valid, x_test_set, y_train, y_valid, _ = load_experiment_from_file(datafile)
    
    if not load_models:
        if train_targets is None:
            models = {}
        else:
            models = pickle.load(open('{}/MLP_models.pkl'.format(destination), 'rb'))
        train_loss = {}
        val_loss = {}
        for t in targets:
            print('########### Training {} #############'.format(t))
            conf = config['hyperparameters'][t]
            model = MLP(conf, x_train.shape[1], y_train[[t]].shape[1], None, None, device=device, relu=False)
            L_t, L_v = model.fit(x_train, y_train[[t]].values, x_valid, y_valid[[t]].values, conf['batch_size'], epochs)
            train_loss[t] = np.array(L_t)
            val_loss[t] = np.array(L_v)
            models[t] = model
        loss = {}
        loss['train'] = train_loss
        loss['validation'] = val_loss
    else:
        models = pickle.load(open('{}/MLP_models.pkl'.format(destination), 'rb'))

    if not load_models:
        print('saving models')
        pickle.dump(models, open('{}/MLP_models.pkl'.format(destination), 'wb'))

    targets = config['hyperparameters'].keys()
    all_predictions = {}
    for sims in x_test_set.keys():
        pred_dict = {}
        for t in targets:
            pred_dict[t] = models[t].predict(x_test_set[sims]).flatten()
        prediction = pd.DataFrame.from_dict(pred_dict)
        all_predictions[sims] = prediction

    if save_results:
        save_prediction_to_file('{}/MLP_prediction.h5'.format(destination), all_predictions)
        if not load_models:
            save_training_error_to_file('{}/MLP_training_error.h5'.format(destination), loss)
    
    return all_predictions

if __name__ == '__main__':
    
    epochs = 200

    parser = argparse.ArgumentParser(description='Raw data parser')
    parser.add_argument('--config_experiment', '-c',
                        dest='experimentconfig',
                        metavar='string',
                        help='path to model config file',
                        default='')

    parser.add_argument('--load_model', '-l',
                        dest='loadmodel',
                        metavar='bool',
                        type=bool,
                        const=1,
                        nargs='?',
                        help='load existing model instead of training new one',
                        default=False)

    parser.add_argument('--epochs', '-e',
                        dest='epochs',
                        metavar='int',
                        nargs='?',
                        const=1,
                        type=int,
                        help='Number of epochs used to train the model',
                        default=100)

    args = parser.parse_args()
    
    if args.experimentconfig == '':
        print('Please provide an experiment configfile.')
        exit(1)
    if args.epochs < 1:
        print('Please provide a positive number of epochs')
        exit(1)

    configfile = args.experimentconfig
    loadm = args.loadmodel
    epochs = args.epochs

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['dataconfig'], 'r') as f:
        dataconfig = yaml.safe_load(f)
    
    with open(config['modelconfig'], 'r') as f:
        modelconfig = yaml.safe_load(f)

    for sp in dataconfig['species']:
        experiment_data = 'data/experiment/{}/{}.h5'.format(dataconfig['experiment_name'], sp)
        save_results_to = '{}/{}/{}'.format(config['destination_folder'], config['experiment_name'], sp)
        train_mlp(modelconfig, experiment_data, save_results_to, epochs, load_models=loadm)
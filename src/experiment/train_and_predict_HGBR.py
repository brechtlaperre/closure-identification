import sys
import argparse

import pandas as pd
import yaml
import pickle

sys.path.append('.')

from src.model.HGBR import HGBR
from src.utils.experiment_utils import load_experiment_from_file, save_prediction_to_file, create_folder

def train_hgbr(config, datafile, destination):

    create_folder(destination)

    targets = config['hyperparameters'].keys()

    x_train, _, x_test_set, y_train, _, _ = load_experiment_from_file(datafile)
    models = {}
    print(x_train.shape)
    for t in targets:
        conf = config['hyperparameters'][t]
        print(conf)
        model = HGBR(**conf, input_scaler=None, output_scaler=None)
        print(y_train[[t]].head())
        model.fit(x_train, y_train[[t]].values)
        models[t] = model

    all_predictions = {}
    for sims in x_test_set.keys():
        pred_dict = {}
        for t in targets:
            pred_dict[t] = models[t].predict(x_test_set[sims]).flatten()
        prediction = pd.DataFrame.from_dict(pred_dict)
        all_predictions[sims] = prediction
    
    pickle.dump(models, open('{}/model.pkl'.format(destination), 'wb'))
    save_prediction_to_file('{}/prediction.h5'.format(destination), all_predictions)

    return all_predictions

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Raw data parser')
    parser.add_argument('--config_experiment', '-c',
                        dest='experimentconfig',
                        metavar='string',
                        help='path to model config file',
                        default='')

    args = parser.parse_args()

    if args.experimentconfig == '':
        print('Please provide an experiment configfile.')
        exit(1)

    configfile = args.experimentconfig

    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['dataconfig'], 'r') as f:
        dataconfig = yaml.safe_load(f)
    
    with open(config['modelconfig'], 'r') as f:
        modelconfig = yaml.safe_load(f)

    for sp in dataconfig['species']:
        experiment_data = 'data/experiment/{}/{}.h5'.format(dataconfig['experiment_name'], sp)
        save_results_to = '{}/{}/{}'.format(config['destination_folder'], config['experiment_name'], sp)
        train_hgbr(modelconfig, experiment_data, save_results_to)
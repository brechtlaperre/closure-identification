import sys

import pandas as pd
import yaml
import pickle
import argparse
sys.path.append('.')

from sklearn.linear_model import LinearRegression
from utils import save_prediction_to_file, create_folder, load_experiment_from_file

def train_lr(modelconfig, datafile, destination):
    
    create_folder(destination)
    
    x_train, _, x_test_set, y_train, _, _ = load_experiment_from_file(datafile)

    lr = LinearRegression(**modelconfig['hyperparameters'])
    lr.fit(x_train, y_train)
    
    all_predictions = {}
    for sims in x_test_set.keys():
        pred_dict = lr.predict(x_test_set[sims])
        prediction = pd.DataFrame(pred_dict, columns = y_train.columns)
        all_predictions[sims] = prediction

    pickle.dump(lr, open('{}/model.pkl'.format(destination), 'wb'))
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
        pred = train_lr(modelconfig, experiment_data, save_results_to)

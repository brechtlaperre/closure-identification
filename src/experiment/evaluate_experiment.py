import sys
import h5py
import pandas as pd
import numpy as np
import pickle
import argparse
import yaml

sys.path.append('.')

from src.utils.data_utils import reverse_transform
from src.utils.experiment_utils import create_folder, load_prediction
from src.utils.evaluate import evaluate
   
def invert_transformation_and_evaluate(pred, truth, transformer):

    truth = reverse_transform(truth, transformer)
    pred = reverse_transform(pred, transformer)
    truth = truth.iloc[pred.index.values].copy()

    #if pred.isin([np.inf, -np.inf]).any(axis=1).sum() > 0:
        #print(pred[pred.isin([np.inf, -np.inf]).any(axis=1)])
        #print(pred[pred.isin([np.inf, -np.inf]).any().values])
    
    #print(pred.isin([np.inf, -np.inf]).sum(axis=0))
    #print(pred.isnull().any(axis=1))
    #print('{} rows were dropped because they were out of boxcox range'.format(truth.shape[0] - pred.shape[0]))

    results = pd.DataFrame.from_dict(evaluate(pred, truth))
    results['feature'] = pred.columns
    results = results.set_index('feature')
    return results

def save_eval_to_latex(data, targets, destination):
    tl = data.set_index(['Type', 'feature', 'model']).stack().unstack([2,3])
    with open(destination, 'w') as f:
        for key in targets:
            ss = tl.iloc[tl.index.get_level_values('Type') == key].copy()
            ss = ss.droplevel('Type')
            f.write('{}\n'.format(key))
            f.write(ss.to_latex(float_format="%.3g"))
            f.write('\n')

def provide_best_metric(data):
    tl = data.set_index(['Type', 'feature', 'model']).stack().unstack([2,3])
    tl['rmse_min'] = tl.iloc[:, tl.columns.get_level_values(1)=='NRMSE [%]'].idxmin(axis=1)
    tl['r2_max'] = tl.iloc[:, tl.columns.get_level_values(1)=='r2'].idxmax(axis=1)
    tl['pr_max'] = tl.iloc[:, tl.columns.get_level_values(1)=='pr'].idxmax(axis=1)
    print(tl[['rmse_min', 'r2_max', 'pr_max']])

def evaluate_experiment(experiment_data, prediction_data, model_name):
    

    #print('################# {} #################'.format(exp))
    pred, truth = load_prediction(prediction_data, experiment_data)
    result_sets = []
    for key in pred.keys():
        #print('################# {} #################'.format(key))
        results = pd.DataFrame.from_dict(evaluate(pred[key], truth[key]))
        
        rmse = invert_transformation_and_evaluate(pred[key], truth[key], transformer)
        subset = rmse[['NRMSE']].copy().rename(columns={'NRMSE': 'NRMSE [%]'}) # #subset = rmse[['RMSE']].copy()
        
        subset['r2'] = results['r2'].values
        subset['pr'] = results['pr'].values
        subset['model'] = model_name
        subset['Type'] = key
        subset = subset.reset_index()
        result_sets.append(subset)
    results = pd.concat(result_sets)
    
    
    results.to_pickle(save_to + 'results.df')
    save_eval_to_latex(results, list(pred.keys()), save_to + 'results_table.txt')    


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
    
    for sp in dataconfig['species']:
        experiment_data = 'data/experiment/{}/{}.h5'.format(dataconfig['experiment_name'], sp)
        prediction_data = '{}/{}/{}/prediction.h5'.format(config['destination_folder'], config['experiment_name'], sp)
        transformer = 'data/experiment/{}/outputtr_{}.pkl'.format(dataconfig['experiment_name'], sp)
        
        save_to = 'results/{}/{}'.format(dataconfig['experiment_name'], config['experiment_name'])
        create_folder(save_to)
        save_to += '/'

        evaluate_experiment(experiment_data, prediction_data, config['model'])
                
    

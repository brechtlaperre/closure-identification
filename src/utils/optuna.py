import os.path
import yaml
from datetime import datetime
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def choose_data_processing(trial, num_features):
    
    input_trans = trial.suggest_categorical('input_trans', ['StandardScaler', 'MinMaxScaler'])
    output_trans = trial.suggest_categorical('output_trans', ['StandardScaler', 'MinMaxScaler'])

    PCA_lvl = trial.suggest_int('PCA_lvl', 1, num_features-1)

    inpipe = create_pipelines({input_trans: [], 'PCA': [PCA_lvl]})
    outpipe = create_pipelines({output_trans: []})

    return inpipe, outpipe

def create_pipelines(attrs):
    inp = []
    for k in attrs.keys():
        if str(k) == 'PCA':
            inp.append(PCA(*attrs[k]))
        else:
            inp.append(getattr(preprocessing, k)(*attrs[k]))

    return make_pipeline(*inp)

def store_optuna_results(modeltype: str, name: str, target, features: list, trainingfile: str, res: dict, params: dict, n_trials: int, timeout: int, target_feat: str, saved_model: str = None) -> dict:
    config = {}
    config['model'] = {'name': name, 'type': modeltype, 'info': '', 'location': saved_model}
    config['target'] = target if type(target) == list else [target] # only store targets as list
    if features is not None:
        config['features'] = features if type(features) == list else [features]
    else:
        config['features'] = None
    config['hyperparameters'] = params
    config['data'] = {'trainingfile' :trainingfile}

    now = datetime.now()
    config['misc'] = dict()
    config['misc']['experiment_date'] = now.strftime("%B %d %Y, %H:%M")
    opt = {'data': trainingfile, 
           'n_trials' : n_trials, 
           'timeout' : timeout, 
           'target feature': target_feat
          }
    if res is not None:
        opt['performance'] = {i: float(j) for i, j in zip(target, res['r2'])}

    config['optuna'] = opt

    if type(target) == list:
        num = 0
        path = 'experiments/{}_{}_{}_{}.yaml'.format(modeltype, name, now.strftime('%b_%d'), num)
        while os.path.isfile(path):
            num += 1
            path = 'experiments/{}_{}_{}_{}.yaml'.format(modeltype, name, now.strftime('%b_%d'), num)
    else:
        num = 0
        path = 'experiments/{}_{}_{}_{}_{}.yaml'.format(modeltype, name, target, now.strftime('%b_%d'), num)
        while os.path.isfile(path):
            num += 1
            path = 'experiments/{}_{}_{}_{}_{}.yaml'.format(modeltype, name, target, now.strftime('%b_%d'), num)


    with open(path, 'w') as file:
        yaml.dump(config, file)

    return config, path

def get_best_params(study):
    params = study.best_params
    if study.best_trial.user_attrs:
        params = {**params, **study.best_trial.user_attrs}

    if study.user_attrs:
        params = {**params, **study.user_attrs}

    return params
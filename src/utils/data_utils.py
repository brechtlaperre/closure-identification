from typing import NamedTuple
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pickle

class DataSet(NamedTuple):
    x: pd.DataFrame
    y: pd.DataFrame
    bin_value: pd.DataFrame = None
    agyro: pd.DataFrame = None

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
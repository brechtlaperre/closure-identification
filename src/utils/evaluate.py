'''Script with metrics for evaluating the model predictions
@author: Brecht Laperre
'''

import torch
from pandas import DataFrame
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, iqr


def evaluate(prediction, target, weights=None, verbal=False) -> dict:
    '''Evaluate prediction with all the defined metrics and return dictionary with results'''
    
    if type(prediction) == DataFrame:
        prediction = prediction.to_numpy()

    if type(target) == DataFrame:
        target = target.to_numpy()

    if type(prediction) == torch.Tensor:
        prediction = prediction.numpy()
    
    if type(target) == torch.Tensor:
        target = target.numpy()

    if target.ndim == 1:
        target = target[:, np.newaxis]
        prediction = prediction[:, np.newaxis]

    try:
        offset, slope, _, _ = linear_relation(prediction, target)
        #offset, slope = None, None
    except ValueError:
        offset, slope = None, None

    pr = pearson_correlation(prediction, target)
    r2 = r2_score(target, prediction, sample_weight=weights, multioutput='raw_values')
    rmse = rmse_score(prediction, target, weights)
    mae = mae_score(prediction, target, weights)
    msa = median_symmetric_accuracy(prediction, target) #mean_error_score(prediction, target)
    me = mean_error_score(prediction, target)
    #PE = prediction_efficiency(prediction, target) # Identical to r2 score
    P_diff = spread_difference(prediction, target)
    P_ratio = spread_ratio(prediction, target)
    yi = modeling_yield(prediction, target)
    rmseiqr = RMSDIQR(prediction, target, weights)
    sspb = symmetric_signed_percentage_bias(prediction, target)

    if verbal:
        print('A: {}, B: {}'.format(offset, slope))
        print('RMSE: {}'.format(rmse))
        print('PR: {}'.format(pr))
        print('R2/PE: {}'.format(r2))
        print('RMSE_IQR: {]'.format(rmseiqr))
        print('MAE: {}'.format(mae))
        print('ME: {}'.format(me))

    res = {'offset': offset, 'slope': slope, 
           'RMSE' : rmse, 'r2' : r2, 'pr': pr, 
           'MAE': mae, 'P_diff': P_diff, 'YI' : yi, 
           'P_ratio': P_ratio, 'NRMSE': rmseiqr, 
           'MSA': msa, 'SSPB': sspb}
    
    return res

def RMSDIQR(M, O, weights):
    '''Root mean square relative error over the interquantile range Q1 - Q3
    Should be more robust against outliers'''
    error = rmse_score(M, O, weights)
    # compute interquartile range Q3 - Q1
    iqrange = iqr(O, rng=(10,90))
    return error/iqrange*100


def symetric_mean_absolute_percentage_error(M, O):
    '''Symmetric mean absolute percentage error.
    Replacement for the MAE, as MAE is heavenly influenced by the largest values in M and O.
    This measure uses a relative error, so independent of the order of magnitude, the over/underfit is found.
    Returns a relative error. Note that because of the positive definite nature of the absolute value,
    the distribution of the error is skewed to the right.
    Liemohn et al. 2021
    '''
    N = M.shape[0]
    return 100*1/N*np.sum(np.abs((O-M)/((O+M)/2)), axis=0)

def symmetric_signed_percentage_bias(M, O):
    '''Symmetric Signed Percentage Bias (SSPB)
    Underprediction will give a negative value, and overprediction will give a positive value; 
    an unbiased forecast will yield 0. 
    This symmetry about zero then mirrors the more common measures of bias, the mean error, and mean percentage error. 
    Due to the log transform, the choice of base affects the result and will determine the level of interpretability for any given data set. 
    We therefore present a new measure of bias based on the log accuracy ratio. 
    '''
    return 100 * np.sign(np.median(np.log(O/M), axis=0))*(np.exp(np.abs(np.median(np.log(O/M), axis=0)))-1)

def median_symmetric_accuracy(M, O):
    '''Median Symmetric Accuracy (MSA)
    More robust version of Mean absolute error.
    Returns an error percentage
    '''
    m = np.median(np.abs(np.log(O/M)), axis=0)
    return 100*(np.exp(m)-1)

def pearson_correlation(M, O):
    res = []
    for i in range(M.shape[1]):
        res.append(pearsonr(M[:,i], O[:, i])[0])
    return np.array(res)

def spread_difference(M, O):
    '''Evaluate how the model predicts the spread of the data.
    A negative number indicates the model underpredicts the spread of data.
    A positive number indicates the model overpredicts the spread of data.
    Perfect value is 0.
    '''
    return np.std(M, axis=0) - np.std(O, axis=0)

def spread_ratio(M, O):
    return np.std(M, axis=0) / np.std(O, axis=0)

def modeling_yield(M, O):
    return (np.max(M, axis=0) - np.min(M, axis=0)) / (np.max(O, axis=0) - np.min(O, axis=0))

def linear_relation(M, O):
    '''Compute linear relation Truth = offset + slope * Prediction
    Input:
        M: Model prediction, shape (batches, time_forward)
        O: observational data
    Output:
        offset
        slope
        sigma_o: standard deviation offset
        sigma_s: standard deviation slope
    '''
    num_examples = M.shape[0]
    delta = num_examples*np.sum(O**2, axis=0) - np.sum(O, axis=0)**2

    try:
        offset, slope = [], []
        for i in range(M.shape[1]):
            res = np.polynomial.Polynomial.fit(M[:, i], O[:, i], deg=1)
            offset.append(res.convert().coef[0])
            slope.append(res.convert().coef[1])

    except:
        print('No convergence')

    sigma_m = np.std(M, axis=0)

    sigma_o = sigma_m * np.sqrt(np.sum(O**2, axis=0) / delta)
    sigma_s = sigma_m * np.sqrt(num_examples/delta)
    

    return np.array(offset), np.array(slope), sigma_o, sigma_s

def rmse_score(M, O, weights):
    return np.sqrt(mean_squared_error(O, M, sample_weight=weights, multioutput='raw_values'))

def mae_score(M, O, weights):
    return mean_absolute_error(O, M, sample_weight=weights, multioutput='raw_values')


def mean_error_score(M, O):
    return np.sum(M - O, axis=0) / M.shape[0]


def prediction_efficiency(M, O):
    return 1 - np.sum((M - O)**2, axis=0) / np.sum((O - np.mean(O, axis=0))**2, axis=0)



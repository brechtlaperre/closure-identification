''''
@author: Brecht Laperre

This script takes the preprocessed data and creates training, validation and test sets 
by sampling data based on the agyrotropy. No transformations of the data are done.
The resulting datasets are stored in the folder data/sampled

'''
import numpy as np
import pandas as pd
import h5py

from preprocess_raw_data import read_preprocessed_data
from src.data.data_utils import DataSet

def sample_on_agyro(data, bins, samplesize, testdata=False, random_state=12345):
    ''' Sample data based on agyrotropy
    Return dataset of size bins*samplesize
    When sampling, replacement is True, meaning small bins are evenly represented as large ones
    '''
    # Drop data with very high agyrotropy
    # data = data[data['Agyro'] < 0.6].copy()
    
    #bins = np.append([data['Agyro'].min()], bins)
    #bins = np.append(bins, [data['Agyro'].max()])

    if type(bins) is int:
        binnames = np.arange(bins)
    else:
        binnames = np.arange(len(bins)-1)

    # Bin data based on agyro
    data['bins'] = pd.cut(data['Agyro'], bins=bins, labels=binnames)
    
    # Sample from each bin
    sets = []
    for i in binnames:
        if data[data.bins == i].shape[0] < samplesize:
            set = data[data.bins == i].sample(samplesize, replace=True, random_state=random_state)
        else:
            set = data[data.bins == i].sample(samplesize, replace=False, random_state=random_state)
        if testdata:
            set = set.drop_duplicates()
        sets.append(set)
    
    return pd.concat(sets, ignore_index=True), binnames

def create_sampled_training_set(fields, species, bins, binsampletrain, binsamplevalid, binsampletest):
    '''Create predefined training, validation and testset based on experiment
    '''
    train_list = ['10000', '13000', '15000', '18000']
    valid_list = ['12000', '17000']
    test_list = ['11000', '14000', '19000']
    with h5py.File('data/processed/processed_data_{}.h5'.format(fields), 'r') as f:
        traindata = []
        for shot in train_list:
            data, dims, _ = read_preprocessed_data(fields, shot, species, f)
            new_data = pd.DataFrame()
            for c in data.columns:
                new_data[c] = data[c].values.reshape(dims)[175:325, 0:400].flatten()
            traindata.append(new_data)

        validdata = []
        for shot in valid_list:
            data, dims, _ = read_preprocessed_data(fields, shot, species, f)
            new_data = pd.DataFrame()
            for c in data.columns:
                new_data[c] = data[c].values.reshape(dims)[175:325, 0:400].flatten()
            data = new_data
            validdata.append(data)

        testdata = []
        for shot in test_list:
            data, dims, _ = read_preprocessed_data(fields, shot, species, f)
            new_data = pd.DataFrame()
            for c in data.columns:
                new_data[c] = data[c].values.reshape(dims)[175:325, 0:400].flatten()
            data = new_data
            testdata.append(data)
        
    traindata = pd.concat(traindata, ignore_index=True)
    validdata = pd.concat(validdata, ignore_index=True)
    testdata = pd.concat(testdata, ignore_index=True)

    trainsample, _ = sample_on_agyro(traindata, bins, binsampletrain)
    validsample, _ = sample_on_agyro(validdata, bins, binsamplevalid)
    testsample, binnames = sample_on_agyro(testdata, bins, binsampletest, testdata=False)

    return trainsample, validsample, testsample, dims, binnames

def create_experiment_from_processed_data(target, features, species, fields, bins=8):
    '''Create a dataset from the processed data.
    Takes a while because of the size of the processed filesize
    '''
    trainsample, validsample, testsample, dims = create_sampled_training_set(fields, species, bins)
    
    x_train = trainsample.loc[:, features].to_numpy()
    y_train = trainsample.loc[:, target].to_numpy()
    
    x_valid = validsample.loc[:, features].to_numpy()
    y_valid = validsample.loc[:, target].to_numpy()
    
    ag_test = None
    x_test = testsample.loc[:, features].to_numpy()
    y_test = testsample.loc[:, target].to_numpy()
    
    if len(target) == 1:
        y_train, y_valid, y_test = map(np.ravel, [y_train, y_valid, y_test])
        
    train_set = DataSet(x=x_train, y=y_train)
    valid_set = DataSet(x=x_valid, y=y_valid)
    test_set = DataSet(x=x_test, y=y_test, agyro=ag_test)

    return train_set, valid_set, test_set, dims, None
    
def load_sample_per_bin(target, features, fname, logscale=None):
    '''Create a dataset from a sample of the processed data
    Data is binned based on agyrotropy and samples are taken from each bin
    '''
    trainsample, validsample, testsample, dims, binnames = read_sampled_dataset_from_file(fname)
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []

    for b in binnames:
        #ag_train = compute_agyro(trainsample)
        x_train.append(trainsample[trainsample['bins'] == b].loc[:, features].to_numpy())
        y_train.append(trainsample[trainsample['bins'] == b].loc[:, target])
        
        x_valid.append(validsample[validsample['bins'] == b].loc[:, features].to_numpy())
        y_valid.append(validsample[validsample['bins'] == b].loc[:, target])
        
        #ag_test = testsample.loc[:, 'Agyro'].to_numpy()
        x_test.append(testsample[testsample['bins'] == b].loc[:, features].to_numpy())
        y_test.append(testsample[testsample['bins'] == b].loc[:, target])

    if logscale:
        for i in range(len(binnames)):
            for j, lg in enumerate(logscale):
                if lg:
                    y_train[i][target[j]] = np.log(y_train[i][target[j]])
                    y_test[i][target[j]] = np.log(y_test[i][target[j]])
                    y_valid[i][target[j]] = np.log(y_valid[i][target[j]])

    y_train = [subset.to_numpy() for subset in y_train]
    y_valid = [subset.to_numpy() for subset in y_valid]
    y_test = [subset.to_numpy() for subset in y_test]

    return x_train, y_train, x_test, y_test, x_valid, y_valid, dims, binnames

def create_experiment_from_stored_sample(target, features, datafiles):
    '''Create a dataset from a sample of the processed data
    Data is binned based on agyrotropy and samples are taken from each bin
    Input:
        target: list
        features: list
        datafiles: list of filenames
    Output:
        train_set -> NamedTuple with x and y for training
        valid_set -> NamedTuple with x and y for validation
        test_set -> NamedTuple with x and y for testing
        dims -> List with grid dimensions
        binnames -> Names of bins used to bin samples
    '''
    if type(datafiles) is not list:
        datafiles = [datafiles]

    xtr, ytr, xv, yv, xt, yt, agt = [], [], [], [], [], [], []

    for files in datafiles:
        trainsample, validsample, testsample, dims, binnames = read_sampled_dataset_from_file(files)

        #ag_train = compute_agyro(trainsample)
        xtr.append(trainsample.loc[:, features])
        ytr.append(trainsample.loc[:, target])
        
        xv.append(validsample.loc[:, features])
        yv.append(validsample.loc[:, target])
        
        agt.append(testsample.loc[:, 'Agyro'])
        xt.append(testsample.loc[:, features])
        yt.append(testsample.loc[:, target])

    x_train, y_train, x_valid, y_valid, x_test, y_test, ag_test = map(pd.concat, [xtr, ytr, xv, yv, xt, yt, agt])

    #if len(target) == 1:
    #    y_train, y_valid, y_test = y_train[:,np.newaxis], y_valid[:,np.newaxis], y_test[:,np.newaxis]
    
    train_set = DataSet(x=x_train, y=y_train)
    valid_set = DataSet(x=x_valid, y=y_valid)
    test_set = DataSet(x=x_test, y=y_test, agyro=ag_test)

    return train_set, valid_set, test_set, dims, binnames

def read_sampled_dataset_from_file(fname):
    '''Read sampled dataset
    The dataset has been created by write_sampled_dataset_to_file
    '''
    with h5py.File(fname, 'r') as f:
        trainset = pd.DataFrame()
        validset = pd.DataFrame()
        testset = pd.DataFrame()
        for key in f['train'].keys():
            trainset[key] = np.array(f['train'][key])
            validset[key] = np.array(f['valid'][key])        
            testset[key] = np.array(f['test'][key])
        dims = f.attrs['Dim']
        bins = f.attrs['bins']
    
    return trainset, validset, testset, dims, bins

def write_sampled_dataset_to_file(fname, fields, species, bins, trainsize, validsize, testsize):

    trainsample, validsample, testsample, dims, binnames = create_sampled_training_set(fields, species, bins, trainsize, validsize, testsize)
    
    trainsample['bins'] = pd.to_numeric(trainsample['bins'])
    validsample['bins'] = pd.to_numeric(validsample['bins'])
    testsample['bins'] = pd.to_numeric(testsample['bins'])
    
    with h5py.File(fname, 'w') as f_r:
        tr_g = f_r.create_group('train')
        val_g = f_r.create_group('valid')
        test_g = f_r.create_group('test')
        f_r.attrs['bins'] = binnames
        f_r.attrs['Species'] = np.string_(species)
        f_r.attrs['Field'] = np.string_(fields)
        f_r.attrs['Dim'] = dims
        for k in trainsample.columns:
            if trainsample[k].dtype == 'object':
                dtype = "S5" # Set to string of fixed length
            else:
                dtype = trainsample[k].dtype
            s = tr_g.create_dataset('{}'.format(k),trainsample[k].shape, dtype=dtype)
            s[...] = trainsample[k].to_numpy().astype(dtype)
            s = val_g.create_dataset('{}'.format(k),validsample[k].shape, dtype=dtype)
            s[...] = validsample[k].to_numpy().astype(dtype)
            s = test_g.create_dataset('{}'.format(k),testsample[k].shape, dtype=dtype)
            s[...] = testsample[k].to_numpy().astype(dtype)
                
def sample_processed_data(datafiles, specie_dict, trspb, vspb, tespb, exp_name = ''):
    '''Prepare dataset for training and validating models
    Creates sampled dataset and stores it to an h5.py file
    Input:
        datafiles: preprocessed file from which to extract the datasets
        specie_dict: dictionary with the species that are considered as key, and as value the name, e.g. {specie: specie_name}
        trspb: Train samples per bin
        vspb: Valid samples per bin
        tespb: Test samples per bin
    Output:
        datafile with path 'data/sampled/{datafile}_{specie_name}_sampled.h5
    '''
    print('Creating samples for {}'.format(filename))
    for specie, name in specie_dict.items():
        
        if len(exp_name) > 0:
            fname = 'data/sampled/{}_{}_{}_sampled.h5'.format(exp_name, filename, name)    
        else:
            fname = 'data/sampled/{}_{}_sampled.h5'.format(filename, name)    
        agyrobins = np.exp(np.array([-np.infty, -3.3, -2.25, -1.5, -0.5]))

        write_sampled_dataset_to_file(fname, filename, specie, agyrobins, trspb, vspb, tespb)

if __name__ == '__main__':

    train_samples_per_bin = 4000 
    valid_samples_per_bin = 1500 
    test_samples_per_bin = 2500 

    datafiles = ['high_res_bg3', 'high_res_bg0', 'high_res_2', 'high_res_hbg']
    specie_dict = {'Species_0+2': 'all_electrons'}

    for filename in datafiles:
        sample_processed_data(filename, specie_dict, train_samples_per_bin, valid_samples_per_bin, test_samples_per_bin)

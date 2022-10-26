# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from implementations import *

def load_data_sample_sub(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "sample-submission.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[0, 1])
    id_ = data[:, 0]
    prediction = data[:, 1]

    return id_, prediction


def load_data_sample(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "train.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1)
    id_ = data [:, 0]
    features = data [:, 2:]
    prediction = np.genfromtxt(
        path_dataset, dtype= str, delimiter=",", skip_header=1, usecols=[1],
        converters={0 :lambda x: 0 if x == 's' else 1})
    
    return id_, features, prediction

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.nanmean(x)
    x = x - mean_x
    std_x = np.nanstd(x)
    x = (x - mean_x) / std_x
    return x, mean_x, std_x

def put_nan(x):
    """Fill the -999.000 values with np.nan"""
    return np.where(x==-999.000, np.nan, x)
    
    
    
def fill_nan(x_nan, median_x):
    """Fill an array x_nan with the median of it"""
    x_filled = np.zeros(x_nan.shape)
    for j in range(x_nan.shape[1]) :
        #print("j : ", j)
        x_filled[:,j] = np.where(np.isnan(x_nan[:,j]), median_x[j], x_nan[:,j])
        #print(x_tr_filled[:,j])
    return x_filled


def search_gamma(y, x, lambda_, initial_w, max_iters, fonction_to_optimize, start_gamma, end_gamma, number) : 
    gamma_tab=np.linspace(start_gamma, end_gamma, number)
    losses_tab=[]
    print(gamma_tab.shape)
    if fonction_to_optimize=='reg_logistic_regression':
        for g in gamma_tab:
            print("gamma = ", g)
            w, losses = reg_logistic_regression(y, x, lambda_, initial_w, max_iters, g)
            losses_tab.append(np.abs(losses))
            print("loss = ", losses)
    if fonction_to_optimize=='mean_squared_error_gd':
        for g in gamma_tab:
            print("gamma = ", g)
            w, losses = mean_squared_error_gd(y, x, initial_w, max_iters, g)
            losses_tab.append(np.abs(losses))
            print("loss = ", losses)
    return gamma_tab, losses_tab
    

def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx



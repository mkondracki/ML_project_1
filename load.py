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
    return (x - np.nanmean(x)) / np.nanstd(x)

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

def fill_nan_value(x_nan, value):
    """Fill an array x_nan with the 0"""
    x_filled = np.zeros(x_nan.shape)
    x_filled = np.where(np.isnan(x_nan), value, x_nan)
    return x_filled

def proportion_of_missing_values(x_tr_nan) : 
    prop = []

    for i in range(x_tr_nan.shape[1]):
        n_missing_value = len(np.where(np.isnan(x_tr_nan[:,i]))[0])
        prop.append(n_missing_value/x_tr_nan[:,i].shape[0])
        
    return np.array(prop)


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
            
    return gamma_tab, losses_tab
    

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly = np.ones((len(x),1))
    for deg in range(1, degree + 1) : 
        poly = np.c_[poly, np.power(x, deg)]

    return poly 



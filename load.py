# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import implementations
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

def standardize_fill_nan(x): 
    x_norm = np.ones(x.shape)
    for k in range(1, x.shape[1]):
        x_norm[:,k] = standardize(x[:,k])
    x_median = np.nanmedian(x_norm, axis=0)
    x_filled = fill_nan(x_norm, x_median)
    return x_filled

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

def proportions_nan (x, threshold): 
    prop_nan = np.empty(x.shape[1])

    for col in range(x.shape[1]): 
        prop_nan[col] = np.sum(np.isnan(x[:,col]))/x.shape[0]
    ind = np.where(prop_nan > threshold) 

    return prop_nan, ind 




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

def reg_log_degree(y_tr, x_tr, degrees, lambda_, gamma, max_iters): 

    losses = []
    ws = []
    for degree in degrees :
        
        loss_temp = []
        w_temp = []
        print("Degree is ", degree)
        
        #create polynomial 
        x_tr_p = build_poly(x_tr, degree)

        #standardization 
        x_tr_p = standardize_fill_nan(x_trp) 
        initial_w = np.full(x_tr_p.shape[1], 0.1)

        loss_temp = np.append(loss_temp, 0)
        loss_temp = np.append(loss_temp, 1)

        i = 0
        while(np.abs(loss_temp[-1]-loss_temp[-2]) > 0.005): 
            print("update w for the ", i+1 , " time")
            i = i+1 
            w, loss = reg_logistic_regression(y_tr, x_tr_p, lambda_,  initial_w, max_iters, gamma)
            initial_w = w[-1]
            w_temp = np.append(w_temp,w[-1])
            loss_temp = np.append(loss_temp,loss[-1])
            
        losses = np.append(losses, loss_temp[-1])
        ws = np.append(ws, w_temp[-1])
                   
    return losses, ws

def search_gamma(y_tr, x_tr, lambda_, initial_w, max_iters, fonction_to_optimize, start_gamma, end_gamma, number) : 
    gamma_tab=np.linspace(start_gamma, end_gamma, number)
    losses_tab=[]
    w_tab = []
    print(gamma_tab.shape)
    if fonction_to_optimize=='reg_logistic_regression':
        for g in gamma_tab:
            print("gamma = ", g)
            w, losses = reg_log_find_gamma(y_tr, x_tr, lambda_,initial_w, g, max_iters)
            w_tab.append(w_tab, w)
            losses_tab.append(losses_tab, losses)
    return gamma_tab, losses_tab

def reg_log_find_gamma(y_tr, x_tr, lambda_,initial_w, gamma, max_iters): 

    losses = []
    ws = []

    loss_temp = []
    w_temp = []
   
    loss_temp = np.append(loss_temp, 0)
    loss_temp = np.append(loss_temp, 1)

    i = 0
    while(np.abs(loss_temp[-1]-loss_temp[-2]) > 0.03): 
        print("update w for the ", i+1 , " time")
        i = i+1 
        w, loss = reg_logistic_regression(y_tr, x_tr, lambda_,  initial_w, max_iters, gamma)
        initial_w = w[-1]
        w_temp = np.append(w_temp,initial_w)
        loss_temp = np.append(loss_temp,loss[-1])

    losses = np.append(losses, loss_temp[-1])
    ws = np.append(ws, w_temp[-1])

    return losses[-1], ws[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    if max_iters > 0:
        for iter in range(max_iters):
            # get loss and update w.
            loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

            ws.append(w)
            losses.append(loss)

            # converge criterion
            if (iter%100)==0 :
                print("iteration : ", iter, " , loss : ", loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    else:
        losses.append(compute_loss(y,tx,initial_w,'negative_log_likelihood'))
            
    return ws,losses




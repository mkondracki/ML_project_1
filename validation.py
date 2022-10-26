#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt


def build_k_indices(y, k_fold, seed):

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def compute_loss(y, tx, w,loss_type):

    loss=0.0
    if loss_type.lower() == 'negative_log_likelihood':
        #compute the cost by negative log likelihood.
        error=sigmoid(np.dot(tx,w))
        loss= np.sum( y@np.log(error)+ (1-y)@np.log(1-error)) / -y.shape[0]
    elif loss_type.lower() == 'mse':  
        # compute loss by MSE

        error=y-tx@w
        loss = (error.T@error)/ (2.0*y.shape[0])
    return loss

def ridge_regression(y, tx, lambda_):
    
    N, D = tx.shape
    lambda_p = 2*N*lambda_
    
    a = tx.T.dot(tx) + lambda_p*np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(a,b)
    return w

def cross_validation(y, x, k_indices, k, lambda_):

    ind_te = k_indices[k]
    ind_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    ind_tr = ind_tr.reshape(-1)

    x_te  = x[ind_te]
    x_tr = x[ind_tr]
    y_te  = y[ind_te]
    y_tr = y[ind_tr]

    w = ridge_regression(y_tr, x_tr , lambda_) 

    loss_tr = np.sqrt(2*compute_loss(y_tr, x_tr, w, 'mse')) 
    loss_te = np.sqrt(2*compute_loss(y_te, x_te, w, 'mse'))

    return loss_tr, loss_te

def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation") 
    
    
def cross_validation_demo(y, x, k_fold, lambdas):

    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over lambdas: TODO
    # ***************************************************
    for ind, lambda_ in enumerate(lambdas):
        rmse_tr_k = []
        rmse_te_k = []
        
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_)
            rmse_tr_k = np.append(rmse_tr_k,loss_tr)
            rmse_te_k = np.append(rmse_te_k,loss_te)
        rmse_tr = np.append(rmse_tr,np.mean(rmse_tr_k))
        rmse_te = np.append(rmse_te,np.mean(rmse_te_k))
        
    index_min = np.argmin(rmse_te)
    best_rmse = rmse_te[index_min]
    best_lambda = lambdas[index_min]
    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
    return best_lambda, best_rmse


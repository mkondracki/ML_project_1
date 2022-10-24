#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import exp

def compute_loss(y, tx, w,loss_type):
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss=0.0
    if loss_type.lower() == 'negative_log_likelihood':
        #compute the cost by negative log likelihood.
        error=sigmoid(np.dot(tx,w))
        loss= np.sum( y*np.log(error)+ (1-y)*np.log(1-error) ) / -y.shape[0]
    elif loss_type.lower() == 'mse':  
        # compute loss by MSE
        #loss = (np.sum((y - np.dot(tx, w))**2)) / (2.0*y.shape[0])
        error=y-np.dot(tx,w)
        loss = (np.dot( np.transpose(error),error))/ (2.0*y.shape[0])
    return loss

def sigmoid(t):
    """
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return  1 / (1 + exp(-t))

def compute_gradient(y, tx, w,regreesion_type):
    """Computes the linear regression or logistic regression gradient at w. 
    Args:
        y: numpy array of shape=(N, ) or shape=(N, 1) in case of logistic regression 
        tx: numpy array of shape=(N,D) or shape=(N, D) in case of logistic regression 
        w: numpy array of shape=(D, ) or shape=(D, 1) in case of logistic regression 
    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
        or 
        a vector of shape (D, 1) in case of logistic regression  
    """
    gradient = []
    # compute linear regression gradient vector
    if loss_type.lower() == 'linear': 
        error = y-np.dot(tx,w) # (N,)
        gradient= (np.dot(np.transpose(tx),error)) / (-1*y.shape[0]) # (2,)
    elif loss_type.lower() == 'logistic':  
        # compute linear regression gradient vector
        error=sigmoid(np.dot(tx,w))-y
        gradient=np.dot(np.transpose(tx),error)/y.shape[0]
    
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute gradient and loss
        gradient = compute_gradient(y,tx,w,'linear')
        loss=compute_loss(y,tx,w,'mse') 
        
        # update w by gradient
        w = w - gamma*gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return ws[-1],losses[-1]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size=1
    
    for n_iter in range(max_iters):

        # implement stochastic gradient descent.
        sum_of_sch_gradient = np.zeros(shape=(2, ))
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            sum_of_sch_gradient=np.add(sum_of_sch_gradient,compute_gradient(minibatch_y,minibatch_tx,w,'linear'))
            
        stochastic_gradient=sum_of_sch_gradient/ batch_size
        loss=compute_loss(y,tx,w,'mse')
        w = w - gamma*stochastic_gradient
        
        ws.append(w)
        losses.append(loss)

    return ws[-1],losses[-1]

def least_squares(y, tx):
    
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    
    #left_part = np.linalg.inv( np.dot( np.transpose(tx), tx))
    #right_part = np.dot(left_part,np.transpose(tx))
    #weights = np.dot(right_part,y)
   
    # weights
    w =np.linalg.solve(np.dot(np.transpose(tx), tx),np.dot(np.transpose(tx), y)) 
   
    # mse loss 
    #inner_part = y-np.dot(tx,w)
    #loss = np.dot( np.transpose(inner_part), inner_part) / (2*y.shape[0])
    loss=compute_loss(y,x,w,'mse')
    return w,loss

def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.  
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: rmse loss
    """
    transposeX=np.transpose(tx)
    xtransx = np.dot(transposeX, tx)
    lamidentity = np.dot(np.identity(xtransx.shape[0]),(lambda_*2*y.shape[0]))
    #weights
    #w = np.dot( np.linalg.inv(xtransx+lamidentity ) ,np.dot(transposeX, y)) 
    w = np.linalg.solve(np.add(xtransx,lamidentity),np.dot(transposeX, y)) 
    loss= np.sqrt(2*compute_loss(y,tx,w,'mse'))
    
    return w,loss

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Implements one step of gradient descent using logistic regression. Return the loss and the updated w.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float
    Returns:
        loss: scalar number
        w: shape=(D, 1) 
    """   
    loss= compute_loss(y,tx,w,'negative_log_likelihood')
    gradient= compute_gradient(y,tx,w,'logistic')
    w_new = w-gamma*gradient
    return loss,w_new

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        ws.append(w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return ws[-1],losses[-1]


def learning_by_penalized_gradient(y, tx, w, gamma,lambda_):
    """
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar numb'/er
        gradient: shape=(D, 1)
    """
    loss = compute_loss(y,tx,w,'negative_log_likelihood')#+lambda_*(np.linalg.norm(w)**2)
    gradient = compute_gradient(y, tx, w,'logistic')+2*lambda_*w
    w_new = w-gamma*gradient
    return loss, w_new

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma,lambda_)
        ws.append(w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return ws[-1],losses[-1]


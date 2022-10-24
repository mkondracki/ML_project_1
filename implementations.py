
#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import exp

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
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
    if regreesion_type.lower() == 'linear': 
        error = y-np.dot(tx,w) # (N,)
        gradient= (np.dot(np.transpose(tx),error)) / (-1*y.shape[0]) # (2,)
    elif regreesion_type.lower() == 'logistic':  
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
    
    if max_iters > 0:
        w = initial_w
        for n_iter in range(max_iters):

            # compute gradient and loss
            gradient = compute_gradient(y,tx,w,'linear')
            # update w by gradient
            w = w - gamma*gradient
            loss=compute_loss(y,tx,w,'mse') 
                # store w and loss
            ws.append(w)
            losses.append(loss)
    else:
        losses.append(compute_loss(y,tx,initial_w,'mse'))
   
        
    return ws[-1],losses[-1][0,0]

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
    
    if max_iters > 0:
        for n_iter in range(max_iters):

            # implement stochastic gradient descent.
            sum_of_sch_gradient = np.zeros(shape=(y.shape[1], ))
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                sum_of_sch_gradient=np.add(sum_of_sch_gradient,compute_gradient(minibatch_y,minibatch_tx,w,'linear'))

            stochastic_gradient=sum_of_sch_gradient/ batch_size         
            w = w - gamma*stochastic_gradient
            loss=compute_loss(y,tx,w,'mse')
            
            ws.append(w)
            losses.append(loss)
    else:
        losses.append(compute_loss(y,tx,initial_w,'mse'))

    return ws[-1],losses[-1][0,0]

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
    loss=compute_loss(y,tx,w,'mse')
    return w,loss[0,0]

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
    loss= compute_loss(y,tx,w,'mse')
    
    return w,loss[0,0]


def logistic_regression(y, tx, initial_w,max_iters, gamma):
    
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    if max_iters > 0:
        for iter in range(max_iters):
            # get loss and update w.
            gradient= compute_gradient(y,tx,w,'logistic')
            w = w-gamma*gradient
            loss= compute_loss(y,tx,w,'negative_log_likelihood')

            ws.append(w)
            losses.append(loss)

            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    else:
        losses.append(compute_loss(y,tx,initial_w,'negative_log_likelihood'))
            
    return ws[-1],losses[-1]


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
            
            gradient = compute_gradient(y, tx, w,'logistic')+2*lambda_*w
            w = w-gamma*gradient
            loss = compute_loss(y,tx,w,'negative_log_likelihood')#+lambda_*(np.linalg.norm(w)**2)
            
            ws.append(w)
            losses.append(loss)

            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    else:
        losses.append(compute_loss(y,tx,initial_w,'negative_log_likelihood'))
            
    return ws[-1],losses[-1]
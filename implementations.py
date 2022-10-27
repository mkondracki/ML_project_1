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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    poly = np.ones((len(x),1))
    for deg in range(1, degree + 1) : 
        poly = np.c_[poly, np.power(x, deg)]

    return poly         
                    
def sigmoid(t):
    """
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-(t)))

def log_loss(y_n, tx_n, w): 
    
    a = max(tx_n@w, -10)
    b = min(tx_n@w, 10)
    loss_n = y_n*np.log(sigmoid(a)) + (1-y_n)*np.log(1-sigmoid(b))
    return loss_n
                       
def compute_loss_log(y, tx, w ): 
    
    N = y.shape[0]
    loss_n = np.array([log_loss(y[i],tx[i],w) for i in range(N)])
    loss = -np.dot(loss_n, np.ones(N))/N
    return loss


def compute_loss_log(y,tx,w): 
    
    temp = tx@w
    temp_max = temp
    temp_min = temp 
    temp_max[temp < -10 ]= -10 
    temp_min[temp> 10 ]= 10    
    
    loss = np.sum(y*np.log(sigmoid(temp_max))+ (1 - y)*np.log(1 - sigmoid(temp_min))) / (-y.shape[0])
    return loss
                  
def compute_loss(y, tx, w,loss_type):
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss = 0.0
    if loss_type.lower() == "negative_log_likelihood":
        # compute the cost by negative log likelihood.
       
        error = sigmoid(tx@w)
        loss = np.sum(y*np.log(error) + (1 - y)*np.log(1 - error)) / (-y.shape[0])
        #loss = compute_loss_log(y, tx,w)
    elif loss_type.lower() == "mse":
        # compute loss by MSE
        # loss = (np.sum((y - np.dot(tx, w))**2)) / (2.0*y.shape[0])
        error = y - tx @ w
        loss = (error.T @ error) / (2.0 * y.shape[0]) 
    return loss



def compute_gradient(y, tx, w, regresion_type):
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
    if regresion_type.lower() == "linear":
        error = y - np.dot(tx, w)  # (N,)
        gradient = (np.dot(np.transpose(tx), error)) / (-1 * y.shape[0])  # (2,)
    # compute logistic regression gradient vector
    elif regresion_type.lower() == "logistic":
        error = sigmoid(tx @ w) - y
        gradient = (tx.T @ error) / y.shape[0]

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
            gradient = compute_gradient(y, tx, w, "linear")
            # update w by gradient
            w = w - gamma * gradient
            loss = compute_loss(y, tx, w,  "mse")
            # store w and loss
            ws.append(w)
            losses.append(loss)
    else:
        losses.append(compute_loss(y, tx, initial_w, "mse"))

    return ws[-1], losses[-1]



# Lea import 
def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic SubGradient Descent algorithm (SubSGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic subgradient
        max_iters: a scalar denoting the total number of iterations of SubSGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SubSGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SubSGD 
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):

        #implement stochastic subgradient descent.

        #calcul of stochastic gradient 
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            g = compute_gradient(minibatch_y, minibatch_tx, w, "linear")
        
        loss = compute_loss(minibatch_y, minibatch_tx, w, "mse")
        #update w 
        w = w - gamma*g

        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return losses[-1], ws[-1]

"""
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    The Stochastic Gradient Descent algorithm (SGD).

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
    

    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1

    if max_iters > 0:
        for n_iter in range(max_iters):

            # implement stochastic gradient descent.
            sum_of_sch_gradient = np.zeros(shape=(y.shape[1],))
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                sum_of_sch_gradient = np.add(
                    sum_of_sch_gradient,
                    compute_gradient(minibatch_y, minibatch_tx, w, "linear"),
                )

            stochastic_gradient = sum_of_sch_gradient / batch_size
            w = w - gamma * stochastic_gradient
            loss = compute_loss(y, tx, w, "mse")

            ws.append(w)
            losses.append(loss)
    else:
        losses.append(compute_loss(y, tx, initial_w, "mse"))

    return ws[-1], losses[-1]

"""
def least_squares(y, tx):

    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    # left_part = np.linalg.inv( np.dot( np.transpose(tx), tx))
    # right_part = np.dot(left_part,np.transpose(tx))
    # weights = np.dot(right_part,y)

    # weights
    w = np.linalg.inv(tx.T @ tx) @ (tx.T @ y)

    # mse loss
    # inner_part = y-np.dot(tx,w)
    # loss = np.dot( np.transpose(inner_part), inner_part) / (2*y.shape[0])
    loss = compute_loss(y, tx, w, "mse")
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mae loss
    """
    lambda_tilda = lambda_ * 2 * y.shape[0]
    xtransx = tx.T @ tx
    w = np.linalg.inv(xtransx + lambda_tilda @ np.identity(xtransx.shape[0])) @ (
        tx.T @ y
    )
    # lamidentity = np.dot(np.identity(xtransx.shape[0]),(lambda_*2*y.shape[0]))
    # weights
    # w = np.dot( np.linalg.inv(xtransx+lamidentity ) ,np.dot(transposeX, y))
    # w = np.linalg.solve(np.add(xtransx,lamidentity),np.dot(transposeX, y))
    loss = np.sqrt(2 * compute_loss(y, tx, w,"mse"))

    return w, loss


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
    loss = compute_loss(y, tx, w, "negative_log_likelihood")
    gradient = compute_gradient(y, tx, w, "logistic")
    w_new = w - gamma * gradient
    return loss, w_new


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w,gamma)
        ws.append(w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return ws[-1], losses[-1]


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
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
    #loss = compute_loss(y, tx, w,  "negative_log_likelihood")
    loss = compute_loss_log(y, tx, w )
    gradient = compute_gradient(y, tx, w, "logistic") + 2 * lambda_ * w

    w_new = w - gamma * gradient

    return loss, w_new


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    # init parameters
    threshold = 1e-20
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        ws.append(w)

        # converge criterion
        losses.append(loss)
        if (iter%100)==0 :
            print("iteration : ", iter, " , loss : ", loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print(losses[-1])
    return ws, losses


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


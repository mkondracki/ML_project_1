#!/usr/bin/env python
# coding: utf-8


import numpy as np


def remove_low_variance(x, threshold): 

    var = np.var(x, axis = 0)
    index = np.array([i for i, v in enumerate(var) if v > threshold])
    x_new = x[:,index]
    return  var, x_new

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
 
def plot_cost (w, b, cost) : 
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection = '3d')
    ax.plot_surface(w, b, cost)
    ax.set(xlabel = "w", ylabel = "b")
    
def plot_loss_iteration(loss):    
    plt.figure(figsize = (9, 6))
    plt.plot(loss)
    plt.xlabel("Iteration", fontsize = 14)
    plt.ylabel("Loss", fontsize = 14)
    plt.title("Loss vs Iteration", fontsize = 14)
    plt.tight_layout()
    plt.show()
    
#!/usr/bin/env python
# coding: utf-8


import numpy as np


def remove_low_variance(x, threshold): 

    var = np.var(x, axis = 0)
    index = np.array([i for i, v in enumerate(var) if v > threshold])
    x_new = x[:,index]
    return  var, x_new


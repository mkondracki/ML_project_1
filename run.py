# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:05:47 2022

@author: maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import implementations, function, helpers

from implementations import *
from function import *
from helpers import *


#please specify the data_path for the test set
data_path_te = "test.csv"

# load data.
y_te, x_te, ids_te = load_csv_data(data_path_te, sub_sample=False)

#remove the column 22 with discrete values
x_te_class = x_te[:, 22]
x_te_reg = np.delete(x_te, 22, axis=1)

#replace missing values by median 
x_te_reg = replace_nan_by_median(x_te_reg)


#split the data according to PRI_jet_num
x_te_split = np.array([x_te_reg[np.where(x_te_class==0)], x_te_reg[np.where(x_te_class==1)], 
                       x_te_reg[np.where(x_te_class==2)], x_te_reg[np.where(x_te_class==3)]], dtype=object)

y_te_split = np.array([y_te[np.where(x_te_class==0)], y_te[np.where(x_te_class==1)], 
                       y_te[np.where(x_te_class==2)], y_te[np.where(x_te_class==3)]], dtype=object)



#remove useless data
x_te_del = [np.delete(x_te_split[0], [4, 5, 6, 8, 12, 22, 23, 24, 25, 26, 27, 28], axis=1),
            np.delete(x_te_split[1], [4, 5, 6, 12, 25, 26, 27], axis=1), 
            x_te_split[2], x_te_split[3]]




#parameters leading to the best score on AICrowd
best_degree = 10
best_lambda = [0, 0, 0, 0]
#best_lambda = [10**(-4), 0, 10**(-4), 0]
best_max_iter = [20000, 20000, 20000, 20000]
best_gamma = [0.2, 0.2, 0.2, 0.2]



#feature Expension and standardization with the whole dataset
degree = best_degree

x_tr_poly = []
x_te_poly = []
for i in range(len(x_tr_del)):
    x_tr_poly.append(standardize(build_poly(x_tr_del[i], degree)))
    x_te_poly.append(standardize(build_poly(x_te_del[i], degree)))
    #keep one's on the first column
    x_tr_poly[i][:,0]=np.ones(x_tr_poly[i][:,0].shape[0])
    x_te_poly[i][:,0]=np.ones(x_te_poly[i][:,0].shape[0])



#Train on the whole dataset
# Define the parameters of the algorithm.
lambda_tab = best_lambda
max_iters_tab = best_max_iter
gamma_tab = best_gamma #learing rate

#w_tab=np.zeros((4, x_tr_tr_poly[0].shape[1]))
#losses_tab=np.array((0,4))
w_tab = []

for i in range(x_tr_split.shape[0]) :  
    print("DATA_SET : ", i)
    #w_init = params_history[-1][i]
    w_init = np.full(x_tr_poly[i].shape[1], 0.1)
    w, losses = reg_logistic_regression(y_tr_split[i], x_tr_poly[i], lambda_tab[i], w_init, max_iters_tab[i], gamma_tab[i])
    w_tab.append(w)
    
    
#TEST

x_te_id = x_te[:, 22]
ind_0 = np.where(x_te_id==0)[0]
ind_1 = np.where(x_te_id==1)[0]
ind_2 = np.where(x_te_id==2)[0]
ind_3 = np.where(x_te_id==3)[0]


proba=np.zeros(x_te.shape[0])
y_hat=[]

proba[ind_0] = x_te_poly[0]@w_tab[0]
proba[ind_1] = x_te_poly[1]@w_tab[1]
proba[ind_2] = x_te_poly[2]@w_tab[2]
proba[ind_3] = x_te_poly[3]@w_tab[3]

for i in proba:
    if i>0:
        y_hat.append(1)
    else :
        y_hat.append(-1)
        
        
#Create the CSV submission

create_csv_submission(ids_te, y_hat, 'submissionfinalX')
















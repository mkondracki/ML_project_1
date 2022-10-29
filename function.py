import numpy as np
import matplotlib.pyplot as plt
import implementations
from implementations import *

# ***************************************************
# Load data
# ***************************************************

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


# ***************************************************
# Preprocessing of the data
# ***************************************************

def put_nan(x):
    """Fill the -999.000 values with np.nan"""
    return np.where(x==-999.000, np.nan, x)

def standardize(x):
    """Standardize the original data set."""
    return (x - np.nanmean(x)) / np.nanstd(x)

def fill_nan(x_nan, median_x):
    """Fill an array x_nan with the median of it"""
    x_filled = np.zeros(x_nan.shape)
    for j in range(x_nan.shape[1]) :
        x_filled[:,j] = np.where(np.isnan(x_nan[:,j]), median_x[j], x_nan[:,j])
    return x_filled

def standardize_fill_nan(x): 
    x_norm = np.ones(x.shape)
    for k in range(1, x.shape[1]):
        x_norm[:,k] = standardize(x[:,k])
    x_median = np.nanmedian(x_norm, axis=0)
    x_filled = fill_nan(x_norm, x_median)
    return x_filled

def fill_nan_value(x_nan, value):
    """Fill an array x_nan with the 0"""
    x_filled = np.zeros(x_nan.shape)
    x_filled = np.where(np.isnan(x_nan), value, x_nan)
    return x_filled

def proportions_nan (x, threshold): 
    """Calculate the proportion of nan values for each columns and the indices 
    of the columns that have a porportion above a threshold."""
    prop_nan = np.empty(x.shape[1])
    for col in range(x.shape[1]): 
        prop_nan[col] = np.sum(np.isnan(x[:,col]))/x.shape[0]
    ind = np.where(prop_nan > threshold) 
    return prop_nan, ind 

def remove_low_variance(x, threshold): 

    var = np.var(x, axis = 0)
    index = np.array([i for i, v in enumerate(var) if v > threshold])
    x_new = x[:,index]
    return  var, x_new

def train_test_split(y, x, ratio, seed ) : 
    """Split the data in order to obtain training and testing set
    The data are splits according to the ratio ( as a percentage of desired size of testing set
    ex : ratio = 0.2 => 20% of testing set."""
    num_row = y.shape[0]
    nb_index_test = int(ratio*num_row)

    np.random.seed(seed)
    indices = np.random.permutation(num_row)

    x_te = x[indices[:nb_index_test]]
    x_tr = x[indices[nb_index_test:]]
    y_te = y[indices[:nb_index_test]]
    y_tr = y[indices[nb_index_test:]]
    
    return y_tr, y_te, x_tr, x_te

# ***************************************************
# Feature expansion 
# ***************************************************
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree """
    poly = np.ones((len(x),1))
    for deg in range(1, degree + 1) : 
        poly = np.c_[poly, np.power(x, deg)]
    return poly 

# ***************************************************
# Regularized logistic regression additional function
# ***************************************************

def reg_log_degree(y_tr, x_tr, degrees, lambda_, gamma, max_iters): 
    """Calculate the logistic regression for a specific lambda and gamma 
    and different degrees of x_tr. 
    While loss is not stabilize continue the process
    Returns : 
        ws : array, contains the best w result of each degree
        losses : array, contains the best loss of each degree
    """
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
                   
    return ws, losses

def reg_log_find_gamma(y_tr, x_tr, lambda_,initial_w, gamma, max_iters): 
    
    """Calculate the logistic regression for a specific lambda and gamma  
    While loss is not stabilize continue the process
    Returns : 
        ws : array, contains the best w result of each degree
        losses : array, contains the best loss of each degree
    """
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

# ***************************************************
# Search gamma 
# ***************************************************
      
def search_gamma(y_tr, x_tr, lambda_, initial_w, max_iters, fonction_to_optimize, start_gamma, end_gamma, number) :
    """Search best gamma for a fonction 

    Returns : 

    gamma_tab : array, contains the gamma that have been test 

    losses_tab : array, contains the corresponding loss.
    """
    gamma_tab=np.linspace(start_gamma, end_gamma, number)
    losses_tab=[]
    w_tab = []
    print(gamma_tab.shape)
    if fonction_to_optimize=='reg_logistic_regression':
        for g in gamma_tab:
            print("gamma = ", g)
            w, losses = reg_log_find_gamma(y_tr, x_tr, lambda_,initial_w, g, max_iters)
            w_tab.append(w)
            losses_tab.append(losses)
    return gamma_tab, losses_tab

# ***************************************************
# Cross validation
# ***************************************************

def build_k_indices(y, k_fold, seed):
    
    """Build k indices for k_fold
    Returns :             
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, gamma):

    max_iters = 1000
    
    ind_te = k_indices[k]
    ind_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    ind_tr = ind_tr.reshape(-1)

    x_te  = x[ind_te]
    x_tr = x[ind_tr]
    y_te  = y[ind_te]
    y_tr = y[ind_tr]
    
    initial_w = np.full(x_tr.shape[1], 0.1)
    # ridge regression 
    ws,losses = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
    #initial_w = ws[-1]
    # calculate the loss for train and test data
    loss_tr =  compute_loss(y_tr, x_tr ,ws, 'negative_log_likelihood')
    loss_te =  compute_loss(y_te, x_te ,ws, 'negative_log_likelihood')

    return loss_tr, loss_te

def cross_validation_lambdas(y, x,  k_fold, lambdas, gamma): 
    
    seed = 12
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    
    for ind, lambda_ in enumerate(lambdas):
        loss_tr_k = []
        loss_te_k = []
        
        for k in range(k_fold):
            loss_tr_temp, loss_te_temp = cross_validation(y, x, k_indices, k, lambda_, gamma)
            
            loss_tr_k = np.append(loss_tr_k,loss_tr_temp)
            loss_te_k = np.append(loss_te_k,loss_te_temp)
            
        loss_tr = np.append(loss_tr, np.mean(loss_tr_k))
        loss_te = np.append(loss_te, np.mean(loss_te_k))

    index_min = np.argmin(loss_te)
    best_loss = loss_te[index_min]
    best_lambda = lambdas[index_min]
    
    cross_validation_visualization(lambdas, loss_tr, loss_te)
    
    return best_lambda, best_loss

def cross_validation_lambdas_with_set(y_tr, x_tr,y_te, x_te,lambdas, gamma): 
    
    seed = 12
   # k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    initial_w = np.full(x_tr.shape[1], 0.1)
    max_iters = 1000 
    for ind, lambda_ in enumerate(lambdas):
        
        ws,losses = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

        # calculate the loss for train and test data
        loss_tr_temp =  compute_loss(y_tr, x_tr ,ws, 'negative_log_likelihood')
        loss_te_temp =  compute_loss(y_te, x_te ,ws, 'negative_log_likelihood')
        
        print(" Loss of training set for ", lambda_, " is ", loss_tr_temp)
        print(" Loss of testing set for ", lambda_, " is ", loss_te_temp)
        loss_tr = np.append(loss_tr, np.mean(loss_tr_temp))
        loss_te = np.append(loss_te, np.mean(loss_te_temp))
           
    index_min = np.argmin(loss_te)
    best_loss = loss_te[index_min]
    best_lambda = lambdas[index_min]
    print(" The best lambda_ is ", best_lambda, " with a loss of  ", best_loss)
    cross_validation_visualization(lambdas, loss_tr, loss_te)
    
    return best_lambda, best_loss

def cross_validation_lambdas_with_set(y_tr, x_tr,y_te, x_te,lambdas, gamma): 
    
    seed = 12
   # k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    initial_w = np.full(x_tr.shape[1], 0.1)
    max_iters = 1000 
    for ind, lambda_ in enumerate(lambdas):
        
        ws,losses = reg_logistic_regression(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)

        # calculate the loss for train and test data
        loss_tr_temp =  compute_loss(y_tr, x_tr ,ws, 'negative_log_likelihood')
        loss_te_temp =  compute_loss(y_te, x_te ,ws, 'negative_log_likelihood')
        
        loss_tr = np.append(loss_tr, np.mean(loss_tr_temp))
        loss_te = np.append(loss_te, np.mean(loss_te_temp))
           
    index_min = np.argmin(loss_te)
    best_loss = loss_te[index_min]
    best_lambda = lambdas[index_min]
    print(" The best lambda_ is ", best_lambda, " with a loss of  ", best_loss)
    cross_validation_visualization(lambdas, loss_tr, loss_te)
    
    return best_lambda, best_loss

def cross_val_find_lambda_degree(y_tr, x_tr, y_te, x_te,lambdas, gamma, degrees): 
    
    for degree in degrees : 
        
        #polynomial expansion 
        x_tr_p = build_poly(x_tr, degree) 
        x_te_p = build_poly(x_te, degree) 
        
        cross_validation_lambdas_with_set(y_tr, x_tr_p, y_te, x_te_p, lambdas, gamma)
        


# ***************************************************
# Plot
# ***************************************************

def prediction(w0, w1, mean_x, std_x):
    """Get the regression line from the model."""
    x = np.arange(1.2, 2, 0.01)
    x_normalized = (x - mean_x) / std_x
    return x, w0 + w1 * x_normalized


def base_visualization(grid_losses, w0_list, w1_list,
                       mean_x, std_x, height, weight):
    """Base Visualization for both models."""
    w0, w1 = np.meshgrid(w0_list, w1_list)

    fig = plt.figure()

    # plot contourf
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(w0, w1, grid_losses.T, cmap=plt.cm.jet)
    fig.colorbar(cp, ax=ax1)
    ax1.set_xlabel(r'$w_0$')
    ax1.set_ylabel(r'$w_1$')
    # put a marker at the minimum
    loss_star, w0_star, w1_star = get_best_parameters(
        w0_list, w1_list, grid_losses)
    ax1.plot(w0_star, w1_star, marker='*', color='r', markersize=20)

    # plot f(x)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(height, weight, marker=".", color='b', s=5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid()

    return fig


def grid_visualization(grid_losses, w0_list, w1_list,
                       mean_x, std_x, height, weight):
    """Visualize how the trained model looks like under the grid search."""
    fig = base_visualization(
        grid_losses, w0_list, w1_list, mean_x, std_x, height, weight)

    loss_star, w0_star, w1_star = get_best_parameters(
        w0_list, w1_list, grid_losses)
    # plot prediciton
    x, f = prediction(w0_star, w1_star, mean_x, std_x)
    ax2 = fig.get_axes()[2]
    ax2.plot(x, f, 'r')

    return fig

def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("Loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def gradient_descent_visualization(
        gradient_losses, gradient_ws,
        grid_losses, grid_w0, grid_w1,
        mean_x, std_x, height, weight, n_iter=None):
    """Visualize how the loss value changes until n_iter."""
    fig = base_visualization(
        grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)

    ws_to_be_plotted = np.stack(gradient_ws)
    if n_iter is not None:
        ws_to_be_plotted = ws_to_be_plotted[:n_iter]

    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[2]
    ax1.plot(
        ws_to_be_plotted[:, 0], ws_to_be_plotted[:, 1],
        marker='o', color='w', markersize=10)
    pred_x, pred_y = prediction(
        ws_to_be_plotted[-1, 0], ws_to_be_plotted[-1, 1],
        mean_x, std_x)
    ax2.plot(pred_x, pred_y, 'r')

    return fig

def plot_loss_iteration(losses): 
    plt.figure(figsize = (9, 6))
    plt.plot(losses)
    plt.xlabel("Iteration", fontsize = 14)
    plt.ylabel("Losses", fontsize = 14)
    plt.title("Losses vs Iteration", fontsize = 14)
    plt.tight_layout()
    plt.show()

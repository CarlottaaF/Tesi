# Import libraries
import GPy as gpy
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import pickle
import timeit
from scipy.spatial import distance
from scipy.spatial.distance import pdist

save_path = "/data_generation_cartella/GPy_models/sparse_gp.pkl"

np.random.seed(123)

# Import parameters and data
X_train = np.loadtxt('/data_generation_cartella/X_train_h1_16_3.csv', delimiter=',') #(115200,16)
y_train = np.loadtxt('/data_generation_cartella/y_train_h1_16_3.csv', delimiter=',') #(115200,25)
X_test = np.loadtxt('/data_generation_cartella/X_test_h1_16_3.csv', delimiter=',') #(12800,16)
y_test = np.loadtxt('/data_generation_cartella/y_test_h1_16_3.csv', delimiter=',') #(12800,25)

tot_samples_train = 115200
tot_samples_test = 12800
n_datapoints = 25
n_parameters = 16

def create_data_groups(X, Y, total_instances, how_many_per_group, num_groups, n_datapoints, n_parameters):

    total_indices = np.arange(total_instances)
    
    # Seleziona gli indici per X_exp e Y_exp (istanze da sporcare)
    indices_exp = np.random.choice(total_indices, size=how_many_per_group, replace=False)
    remaining_indices = np.setdiff1d(total_indices, indices_exp)
    
    # Array per memorizzare i gruppi
    Y_groups = np.empty((num_groups, how_many_per_group, n_datapoints))
    X_groups = np.empty((num_groups, how_many_per_group, n_parameters))
    
    for i in range(num_groups):
        selected_indices = np.random.choice(remaining_indices, size=how_many_per_group, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, selected_indices)
        
        Y_groups[i] = Y[selected_indices]
        X_groups[i] = X[selected_indices]
    
    Y_exp = Y[indices_exp]
    X_exp = X[indices_exp]
    
    return X_exp, Y_exp, X_groups, Y_groups

how_many_per_group_train = 900
how_many_per_group_test = 100
num_groups = 10

X_exp_train, Y_exp_train, X_train, Y_train = create_data_groups(X_train, y_train, tot_samples_train, how_many_per_group_train, num_groups, n_datapoints, n_parameters)
X_exp_test, Y_exp_test, X_test, Y_test = create_data_groups(X_test, y_test, tot_samples_test, how_many_per_group_test, num_groups, n_datapoints, n_parameters)


# Add noise to the data
noise = 0.01
Y_exp_train = Y_exp_train + np.random.normal(loc=0.0, scale=noise, size=Y_exp_train.shape)  
Y_exp_test = Y_exp_test + np.random.normal(loc=0.0, scale=noise, size=Y_exp_test.shape)  


# Compute the log-likelihood
def compute_loglikelihood(Y_exp, Y, how_many_per_group, n_groups, sigma, n_datapoints):

    log_lik = np.zeros((how_many_per_group, n_groups)) 
    
    for i1 in range(how_many_per_group):
        for i2 in range(n_groups):
                somma = 0
                for i3 in range(n_datapoints):
                    somma += np.log(1. / (np.sqrt(2. * np.pi) * sigma)) + (-(((Y_exp[i1,i3]-Y[i2,i1,i3] ** 2) / (2. * (sigma ** 2)))))
                log_lik[i1, i2] = somma 
    
    return log_lik

log_like_train = compute_loglikelihood(Y_exp_train, Y_train, how_many_per_group_train, num_groups, noise, n_datapoints)
log_like_test = compute_loglikelihood(Y_exp_test, Y_test, how_many_per_group_test, num_groups, noise, n_datapoints)


# Reshape the log-likelihood
def reshape_loglikelihood(log_like, how_many_per_group, n_groups):

    likelihood_r = np.reshape(log_like, (how_many_per_group* n_groups,),'F') # dati riorganizzati per colonne 
    return likelihood_r

log_like_train_r = reshape_loglikelihood(log_like_train, how_many_per_group_train, num_groups)
log_like_test_r = reshape_loglikelihood(log_like_test, how_many_per_group_test, num_groups)

# Normalize the log-likelihood
log_like_train_norm = (log_like_train_r - np.mean(log_like_train_r)) / np.std(log_like_train_r)
log_like_test_norm = (log_like_test_r - np.mean(log_like_test_r)) / np.std(log_like_test_r)

# oppure
# log_like_train_norm = (log_like_train_r - np.min(log_like_train_r)) / (np.max(log_like_train_r) - np.min(log_like_train_r))
# log_like_test_norm = (log_like_test_r - np.min(log_like_test_r)) / (np.max(log_like_test_r) - np.min(log_like_test_r))

# Reshape X_exp

# Reshape Y_exp
Y_exp_train_r = np.tile(Y_exp_train, (num_groups, 1))
Y_exp_test_r = np.tile(Y_exp_test, (num_groups, 1))

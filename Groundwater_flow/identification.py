# Standard Library Imports
import os
import sys
import timeit
from scipy.optimize import least_squares

# Optimize performance by setting environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from timeit import repeat

# Third-party Library Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, multivariate_normal, uniform
from tensorflow.keras.models import load_model
from itertools import product
from numba import jit
import pickle

# Local Module Imports
sys.path.append('/data_generation_cartella/')
import tinyDA as tda

from model import Model

# Save path of the GP model
save_path = "/data_generation_cartella/GPy_models/sparse_gp.pkl"
with open(save_path, "rb") as f:
        m_load = pickle.load(f)  

# Initialize Parameters
n_samples = 25
np.random.seed(123)
random_samples = np.random.randint(0, 160, n_samples)
n_eig = 16
X_values = np.loadtxt('/data_generation_cartella/X_test_h1_16_3.csv', delimiter=',') 
y_values = np.loadtxt('/data_generation_cartella/y_test_h1_16_3.csv', delimiter=',')

# Resolution Parameters for Different Solvers
resolutions = [(50, 50), (25, 25)]
field_mean, field_stdev, lamb_cov, mkl = 1, 1, 0.1, 16 

# Instantiate Models for Different Resolutions
solver_h1 = Model(resolutions[0], field_mean, field_stdev, mkl, lamb_cov)
solver_h2 = Model(resolutions[1], field_mean, field_stdev, mkl, lamb_cov)

def setup_random_process(solver_high, solver_low):
    """
    Synchronize the random processes between the higher and lower fidelity models
    by matching transmissivity fields across different resolutions.
    """
    coords_high = solver_high.solver.mesh.coordinates()
    coords_low = solver_low.solver.mesh.coordinates()
    
    structured_high = np.array(coords_high).view([('', coords_high.dtype)] * coords_high.shape[1])
    structured_low = np.array(coords_low).view([('', coords_low.dtype)] * coords_low.shape[1])
    
    bool_mask = np.in1d(structured_high, structured_low)
    solver_low.random_process.eigenvalues = solver_high.random_process.eigenvalues
    solver_low.random_process.eigenvectors = solver_high.random_process.eigenvectors[bool_mask]  

# Setup random processes between solvers
setup_random_process(solver_h1, solver_h2)

# Define Points for Data Extraction
x_data = y_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
datapoints = np.array(list(product(x_data, y_data)))

# Solver Data Functions
def solver_h1_data(x):
    solver_h1.solve(x)
    return solver_h1.get_data(datapoints)
def solver_h2_data(x):
    solver_h2.solve(x) 
    return solver_h2.get_data(datapoints)

def model_HF(input): return solver_h1_data(input).flatten()
def model_LF1(input): return solver_h2_data(input).flatten()



# MCMC Parameters
noise        = 0.001
scaling1     = 1
n_iter       = 11000
burnin       = 1000
thin         = 1
sub_sampling = [5,5]






class y_distribution_coarse:

    # Aggiungere un set_bias method per adaptive error model
           
    def loglike(self,Y):

        loglike_train = 0 #prendere valore file GPSR.py
        Y_pred, Y_var = m_load.predict(Y) 
        Y_pred_rescaled = Y_pred * np.std(loglike_train) + np.mean(loglike_train)
        #aggiungere parte adattiva

        return Y_pred_rescaled


# Sampling for Each Random Sample
for i, sample in enumerate(random_samples, start=1):
    print(f'Sample = {sample}')
    x_true = X_values[sample]
    y_true = y_values[sample]
    y_observed = y_true + np.random.normal(scale=noise, size=y_true.shape[0])

    # Likelihood Distributions

    cov_likelihood = noise**2 * np.eye(25)
    y_distribution_fine = tda.AdaptiveGaussianLogLike(y_observed, cov_likelihood*scaling1)
    y_distribution_coarse = y_distribution_coarse()

    # Proposal Distribution

    def ls(x):
        return (y_true-model_HF(x))**2

    res = least_squares(ls,np.zeros_like((x_true)), jac='3-point')
    covariance = np.linalg.pinv(res.jac.T @ res.jac)
    covariance *= 1/np.max(np.abs(covariance))

    my_proposal = tda.GaussianRandomWalk(C=covariance,scaling=1e-1, adaptive=True, gamma=1.1, period=10)



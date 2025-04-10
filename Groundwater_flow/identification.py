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
from joblib import load

# Local Module Imports
sys.path.append('/data_generation_cartella/')
import tinyDA as tda

from model import Model

save_path = "/data_generation_cartella/GPy_models/sparse_gp.pkl"
with open(save_path, "rb") as f:
        m_load = pickle.load(f)  
print(m_load)

pca = load('/data_generation_cartella/GPy_models/pca_model_sparse.joblib')

# MCMC Parameters
noise        = 0.01#0.001
scaling1     = 1
n_iter       = 10000
burnin       = 1000
thin         = 1
sub_sampling = 1
case         = "MLDA" #"MLDA" or "1level" 

# Initialize Parameters
n_samples = 1 #rimettere a 25
np.random.seed(123)
random_samples = np.random.randint(0, 12800, n_samples) 
n_eig = 16
X_values = np.loadtxt('/data_generation_cartella/X_test_h1_16_3.csv', delimiter=',')
y_values = np.loadtxt('/data_generation_cartella/y_test_h1_16_3.csv', delimiter=',')
y_values_pca = pca.transform(y_values)


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
def solver_h2_param(x):
    solver_h2.solve(x)
    return solver_h2.parameters 

def model_HF(input): return solver_h1_data(input).flatten()
def model_LF1(input): return solver_h2_param(input).flatten()


# Prior and Proposal Distributions
x_distribution = stats.multivariate_normal(mean=np.zeros(16), cov=np.eye(16))
Times, Time_ESS, ESS, samples_tot, ERR = [], [], [], [], []


# Sampling for Each Random Sample
for i, sample in enumerate(random_samples, start=1):
    print(f'Sample = {sample}')
    x_true = X_values[sample]
    y_true = y_values[sample]
    y_true_pca = y_values_pca[sample]
    y_observed = y_true + np.random.normal(scale=noise, size=y_true.shape[0])
    y_observed_pca = y_true_pca + np.random.normal(scale=noise, size=y_true_pca.shape[0])

    y_observed_pca_r = y_observed_pca.reshape(1,10)

    @jit(nopython=True, fastmath=True)  # JIT per ottimizzare i calcoli
    def prepare_input(params, y_observed_r):
        return np.hstack((y_observed_r, params))  
    

    # Likelihood Distributions

    cov_likelihood = noise**2 * np.eye(25)
    y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)


    # Define the coarse likelihood distribution 
    class y_distribution_coarse:
        def __init__(self, y_observed_pca_r):
            self.y_observed_pca_r = y_observed_pca_r
            
                   
        def loglike(self,params):
            #print(params)
            log_min = -587.9534777849608#-595.9109842692088 #
            log_max = 212.25108202058436  #192.271837913472 #
            params = params.reshape(1,16)
            Input_test = prepare_input(params,self.y_observed_pca_r)
            Y_pred, Y_var = m_load.predict(Input_test)
            Y_var_true = Y_var * (log_max - log_min)**2
            Y_pred_rescaled = (Y_pred * (log_max - log_min) + log_min)
            Y_pred_rescaled = Y_pred_rescaled.item() - 10*np.sqrt(Y_var_true.item())

            return Y_pred_rescaled
        

    y_distribution_coarse = y_distribution_coarse(y_observed_pca_r)

    # Proposal Distribution

    def ls(x):
        return (y_true-model_HF(x))**2

    res = least_squares(ls,np.zeros_like((x_true)), jac='3-point')
    covariance = np.linalg.pinv(res.jac.T @ res.jac)
    covariance *= 1/np.max(np.abs(covariance))

    my_proposal = tda.GaussianRandomWalk(C=covariance,scaling=1e-3, adaptive=True, gamma=1.1, period=10) #gamma=1.1, period=10
 

    # Initialize Posteriors
    my_posteriors = [
        tda.Posterior(x_distribution, y_distribution_coarse, model_LF1), 
        tda.Posterior(x_distribution, y_distribution_fine, model_HF),
    ]if case != "1level" else tda.Posterior(x_distribution, y_distribution_coarse, model_LF1)

    # Run MCMC Sampling
    start_time = timeit.default_timer()
    samples = tda.sample(my_posteriors, my_proposal, iterations=n_iter, n_chains=1,
                          subchain_length=sub_sampling, initial_parameters=res.x,
                            store_coarse_chain=False,force_sequential=True)
    #initial_parameters=res.x
    elapsed_time = timeit.default_timer() - start_time

     # Effective Sample Size (ESS)
    idata = tda.to_inference_data(samples, level='fine').sel(draw=slice(burnin, None, thin), groups="posterior")
    ess = az.ess(idata)
    mean_ess = np.mean([ess.data_vars[f'x{j}'].values for j in range(16)])

    az.plot_trace(idata)
    plt.savefig('/data_generation_cartella/images/DA_02.png', dpi=300, bbox_inches="tight")

    # Store Results
    Times.append(elapsed_time)
    ESS.append(mean_ess)
    Time_ESS.append(elapsed_time / mean_ess)
    post = idata.posterior
    val=post.mean().to_array()
    err=(np.mean(np.sqrt((x_true-val)**2)))
    ERR.append(err)
    print(f'Time: {elapsed_time:.2f}, ESS: {mean_ess:.2f}, Time/ESS: {elapsed_time / mean_ess:.2f}, Err: {err:.3f} ({i}/{n_samples})')


# Save Results
# output_folder = '/data_generation_cartella/recorded_values'
# np.save(os.path.join(output_folder, f'MDA_MF_{case}_ratio_001_coarse.npy'), Time_ESS)
# np.save(os.path.join(output_folder, f'MDA_MF_{case}_times_001_coarse.npy'), Times)
# np.save(os.path.join(output_folder, f'MDA_MF_{case}_err_001_coarse.npy'), ERR)
# np.save(os.path.join(output_folder, f'MDA_MF_{case}_ESS_001_coarse.npy'), ESS)


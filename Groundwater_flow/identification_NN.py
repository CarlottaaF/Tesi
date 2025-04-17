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
from scipy.optimize import minimize
from numpy.linalg import norm
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pickle
import os
from scipy.stats import qmc
import sys
from sklearn.decomposition import PCA

# Local Module Imports
sys.path.append('/data_generation_cartella/')
import tinyDA as tda

from model import Model


Regressor = keras.models.load_model("/data_generation_cartella/NN/model_02.keras",compile=False)

for layer in Regressor.layers:
    layer.trainable = False
Regressor.trainable = False

with open("/data_generation_cartella/NN/hist_02.pkl", 'rb') as file:
    hist = pickle.load(file)

pca = load('/data_generation_cartella/GPy_models/pca_model.joblib')

# MCMC Parameters
noise        = 0.01#0.001
scaling1     = 1
n_iter       = 10000
burnin       = 1000
thin         = 1
sub_sampling = 1
case         = "1level" # "DA" or "1level" 

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
    y_observed_pca_r = y_observed_pca.reshape(1,16)
    
    # LS
    def ls(x):
        return (y_true-model_HF(x))**2

    res = least_squares(ls,np.zeros_like((x_true)), jac='3-point')

    # MAP
    sigma2 = noise**2  
    tau2 = 1.0          # prior variance

    def neg_log_posterior(x):
        likelihood_term = 0.5 / sigma2 * np.sum((y_true - model_HF(x))**2)
        prior_term = 0.5 / tau2 * np.sum(x**2)
        return likelihood_term + prior_term

    res_MAP = minimize(neg_log_posterior, np.zeros_like(x_true), method='BFGS')

    
    # Likelihood Distributions

    cov_likelihood = noise**2 * np.eye(25)
    y_distribution_fine = tda.GaussianLogLike(y_observed, cov_likelihood)

    # Termine correttivo

    log_min = -235.9379257453965
    log_max = -0.6140630795649933

    loglik_FOM = 0.0
    for i in range(25):
        loglik_FOM += -((y_observed[i]-model_HF(res_MAP.x)[i]) ** 2) / (2. * (noise) ** 2)

    res_x = res.x.reshape(1,16)   
    res_MAP = res_MAP.x.reshape(1,16)     
    Pred_NN_MAP = Regressor([y_observed_pca_r, res_MAP], training=False)
    Pred_NN_MAP_true = ((Pred_NN_MAP * (log_max - log_min) + log_min))


    # Converte subito in Tensor
    y_observed_pca_r = tf.constant(y_observed_pca_r, dtype=tf.float32)
    if len(y_observed_pca_r.shape) == 1:
        y_observed_pca_r = tf.expand_dims(y_observed_pca_r, axis=0)
    y_observed_pca_r.set_shape([1, 16])
    


    # Define the coarse likelihood distribution 
    class y_distribution_coarse:
        def __init__(self, y_observed_pca_r):
            self.y_observed_pca_r = y_observed_pca_r

        @tf.function(jit_compile=True,reduce_retracing=True)
        def predict_optimized(self, params):
            return Regressor([self.y_observed_pca_r, params], training=False)
        
        @tf.function(jit_compile=True,reduce_retracing=True)
        def loglike(self, params):
            
            params = tf.reshape(params, (1, 16))
            predictions = self.predict_optimized(params)
            predictions_true = (predictions * (log_max - log_min) + log_min) + loglik_FOM - Pred_NN_MAP_true[0,0]

            return predictions_true[0,0]


        

    y_distribution_coarse = y_distribution_coarse(y_observed_pca_r)

    # Proposal Distribution

    #print(res.x)
    #print(x_true)
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
    #print(az.summary(idata))

    az.plot_trace(idata)
    plt.savefig('/data_generation_cartella/images/prova.png', dpi=300, bbox_inches="tight")

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


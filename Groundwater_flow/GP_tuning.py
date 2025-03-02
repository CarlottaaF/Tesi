# Import libraries
import GPy as gpy
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import pickle
import timeit
from scipy.spatial import distance
import optuna

np.random.seed(123)

# Import parameters and data
X_train = np.loadtxt('/data_generation_cartella/X_train_h1_16_3.csv', delimiter=',') #(115200,16)
y_train = np.loadtxt('/data_generation_cartella/y_train_h1_16_3.csv', delimiter=',') #(115200,25)

n_samples_train = 115200
n_datapoints = 25
n_parameters = 16

# Prendere una parte di questi dati da sporcare e una da lasciare pulita
how_many_train = 10000
indices_noisy_train = np.random.choice(n_samples_train, size=how_many_train, replace=False)
remaining_indices_train = np.setdiff1d(np.arange(n_samples_train), indices_noisy_train)
indices_clean_train = np.random.choice(remaining_indices_train, size=how_many_train, replace=False)

y_clean_train = y_train[indices_clean_train]
y_to_be_noisy_train = y_train[indices_noisy_train]
X_train_new = X_train[indices_noisy_train]

# Add noise to the data
noise = 0.01
y_observed_train = y_to_be_noisy_train + np.random.normal(loc=0.0, scale=noise, size=y_to_be_noisy_train.shape)

# Compute the log-likelihood
log_like_train = np.zeros(how_many_train)

for i in range(how_many_train):
    for j in range(n_datapoints):
        single_value = -0.5 * np.log(2 * np.pi * noise**2) - ((y_observed_train[i,j] - y_clean_train[i,j]) ** 2) / (2 * noise**2)
        log_like_train[i] = log_like_train[i] + single_value

log_like_train = log_like_train.reshape(-1, 1)

# Normalizzare la log-likelihood
log_like_train_norm = (log_like_train - np.mean(log_like_train)) / np.std(log_like_train)

# Definizione del kernel 
kernel = gpy.kern.RBF(input_dim=n_parameters, variance=1.0, lengthscale=1.0) 

# Imposta gli inducing points (campionamento casuale)
n_inducing_points = 600
#Z = X_train_new[np.random.choice(how_many_train, size=n_inducing_points, replace=False), :]
kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=30)
kmeans.fit(X_train_new)
Z = kmeans.cluster_centers_  # Centroidi dei cluster

# Creazione del modello Sparse GP Regression fuori dalla funzione obiettivo
m = gpy.models.SparseGPRegression(X_train_new, log_like_train_norm, kernel=kernel, Z=Z)
print(m)
m.inducing_inputs.fix()  # Blocchiamo gli inducing points

# Funzione obiettivo per Optuna
def objective(trial):
    variance = trial.suggest_loguniform("variance", 0.01, 10)
    lengthscale = trial.suggest_loguniform("lengthscale", 0.1, 10)
    gaussian_noise_variance = trial.suggest_loguniform("gaussian_noise_variance", 1e-6, 1e-2)
    
    # Assegna i valori suggeriti dal trial
    m.rbf.variance = variance
    m.rbf.lengthscale = lengthscale
    m.Gaussian_noise.variance = gaussian_noise_variance
    
    # Ottimizzazione con BFGS
    m.optimize("bfgs")
    
    # Restituisce il valore negativo della log-likelihood (poiché Optuna minimizza per default)
    return -m.log_likelihood()

# Creazione dello studio Optuna
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)  # Numero di iterazioni

# Migliori parametri trovati
best_params = study.best_params
m.rbf.variance = best_params["variance"]
m.rbf.lengthscale = best_params["lengthscale"]
m.Gaussian_noise.variance = best_params["gaussian_noise_variance"]

print("Miglior modello ottimizzato:")
print(m)

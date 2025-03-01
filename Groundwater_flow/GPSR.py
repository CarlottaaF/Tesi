# Import libraries
import GPy as gpy
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import pickle
import timeit
from scipy.spatial import distance

save_path = "/data_generation_cartella/GPy_models/sparse_gp.pkl"

np.random.seed(123)

# Import parameters and data
X_train = np.loadtxt('/data_generation_cartella/X_train_h1_16_3.csv', delimiter=',') #(115200,16)
y_train = np.loadtxt('/data_generation_cartella/y_train_h1_16_3.csv', delimiter=',') #(115200,25)
X_test = np.loadtxt('/data_generation_cartella/X_test_h1_16_3.csv', delimiter=',') #(12800,16)
y_test = np.loadtxt('/data_generation_cartella/y_test_h1_16_3.csv', delimiter=',') #(12800,25)

n_samples_train = 115200
n_samples_test = 12800
n_datapoints = 25
n_parameters = 16

# Prendere una parte di questi dati da sporcare e una da lasciare pulita
# TRAINING SET
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

# TEST SET
how_many_test = 2000
indices_noisy_test = np.random.choice(n_samples_test, size=how_many_test, replace=False)
remaining_indices_test = np.setdiff1d(np.arange(n_samples_test), indices_noisy_test)
indices_clean_test = np.random.choice(remaining_indices_test, size=how_many_test, replace=False)

y_clean_test = y_test[indices_clean_test]
y_to_be_noisy_test = y_test[indices_noisy_test]
X_test_new = X_test[indices_noisy_test]

# Add noise to the data
noise = 0.01
y_observed_test = y_to_be_noisy_test + np.random.normal(loc=0.0, scale=noise, size=y_to_be_noisy_test.shape)

# Compute the log-likelihood
log_like_test = np.zeros(how_many_test)

for i in range(how_many_test):
    for j in range(n_datapoints):
        single_value = -0.5 * np.log(2 * np.pi * noise**2) - ((y_observed_test[i,j] - y_clean_test[i,j]) ** 2) / (2 * noise**2)
        log_like_test[i] = log_like_test[i] + single_value

log_like_test = log_like_test.reshape(-1, 1)
#print(log_like_test)


# IMPOSTARE IL KERNEL PER IL GAUSSIAN PROCESS

# classRBF(input_dim, variance=1.0, lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False
kernel = gpy.kern.RBF(input_dim=n_parameters, variance=1.0, lengthscale=None) + gpy.kern.White(input_dim=n_parameters, variance=0.1)
# aggiungere un white noise?


# SCEGLIERE GLI INDUCING POINTS PER LA SPARSE REGRESSION
n_inducing_points = 600

# Casualmente
Z = X_train_new[np.random.choice(how_many_train, size=n_inducing_points, replace=False), :]

# K-means clustering: centroidi dei cluster come inducing points
#sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
# kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=30)
# kmeans.fit(X_train_new)
# Z = kmeans.cluster_centers_  # Centroidi dei cluster




train_or_test = 1 # 1 for train, 0 for test


if train_or_test == 1:
    
    # ALLENARE IL MODELLO SRGP

    # Creazione del modello di regressione GP sparso
    m = gpy.models.SparseGPRegression(X_train_new, log_like_train_norm, Z=Z)
    print("Modello iniziale:")
    print(m)
    
    start_time = timeit.default_timer()
    
    # **Ottimizzazione 1: fissiamo gli inducing points e ottimizziamo i parametri del kernel**
    m.inducing_inputs.fix()
    best_log_marginal_likelihood = -np.inf
    best_variance = None
    best_lengthscale = None
    best_gaussian_noise_variance = None
    
    # **Multi-start optimization**
    num_restarts = 10

    for _ in range(num_restarts):
        rand_gen = np.random.default_rng() 
        m.rbf.variance = rand_gen.uniform(0.01,10)
        #print(f"Variance: {m.rbf.variance}")
        m.rbf.lengthscale = rand_gen.uniform(0.1,5)
        #print(f"Lengthscale: {m.rbf.lengthscale}")
        m.likelihood.variance = rand_gen.uniform(0.001,1)
        #print(f"Gaussian noise variance: {m.likelihood.variance}")
        m.optimize('bfgs')
        log_marginal_likelihood = m.log_likelihood()
        
        if log_marginal_likelihood > best_log_marginal_likelihood:
            best_log_marginal_likelihood = log_marginal_likelihood
            best_variance = m.rbf.variance
            best_lengthscale = m.rbf.lengthscale
            best_gaussian_noise_variance = m.Gaussian_noise.variance

    # Imposta il modello con i migliori parametri trovati
    if all(x is not None for x in [best_variance, best_lengthscale, best_gaussian_noise_variance]):
        m.rbf.variance = best_variance
        m.rbf.lengthscale = best_lengthscale
        m.Gaussian_noise.variance = best_gaussian_noise_variance

    print(m)


    
    # **Ottimizzazione 2: sblocchiamo gli inducing points e ottimizziamo tutti i parametri**
    m.Z.unconstrain()
    m.optimize('bfgs')
    print(m)

    elapsed_time = timeit.default_timer() - start_time
    print(f"Tempo di esecuzione: {elapsed_time}")

    # SALVARE IL MODELLO

    with open(save_path, "wb") as f:
        pickle.dump(m, f)  #  Salva l'intero modello
else:

    # CARICARE IL MODELLO
    with open(save_path, "rb") as f:
        m_load = pickle.load(f)  #  Carica il modello senza dover reimpostare i parametri

    print(m_load)

    # FARE PREDIZIONE SUL TEST_SET
    Y_pred, Y_var = m_load.predict(X_test_new) 
    Y_pred_rescaled = Y_pred * np.std(log_like_train) + np.mean(log_like_train)
    #print(Y_pred_rescaled)

    # Errore assoluto medio
    mae = np.mean(np.abs(Y_pred_rescaled - log_like_test))
    print(f"Errore Assoluto Medio (MAE): {mae}")
    
    # Plot usando hexbin
    plt.figure(figsize=(6.5, 6.5), dpi=100)
    hb = plt.hexbin(log_like_test, Y_pred_rescaled, gridsize=50, cmap='plasma', mincnt=1)
    cb = plt.colorbar(hb, label='Frequency')
    
    # Linea guida y=x
    lims = [min(log_like_test.min(), Y_pred_rescaled.min()), 
            max(log_like_test.max(), Y_pred_rescaled.max())]
    plt.plot(lims, lims, 'gold', linewidth=1.5, zorder=-1, label='y=x')
    
    # Etichette e legenda
    plt.xlabel('True Value [%]')
    plt.ylabel('Prediction [%]')
    plt.title('Hexbin Density Plot')
    plt.legend(loc='upper left') 

    plt.savefig('/data_generation_cartella/images/parity_plot_SRGP.png', format='png', dpi=300, bbox_inches='tight')




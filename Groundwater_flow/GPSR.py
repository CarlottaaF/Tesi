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

n_samples_train = 115200
n_samples_test = 12800
n_datapoints = 25
n_parameters = 16

# Prendere una parte di questi dati da sporcare e una da lasciare pulita
# TRAINING SET
how_many_train = 5000
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
#log_like_train_norm = (log_like_train - np.min(log_like_train)) / (np.max(log_like_train) - np.min(log_like_train))

# TEST SET
how_many_test = 500
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

# Stima del lengthscale prima della selezione degli inducing points
pairwise_dists = pdist(X_train_new, metric='euclidean')
lengthscale_init = np.median(pairwise_dists)

# Kernel RBF
kernel = gpy.kern.RBF(input_dim=n_parameters, variance=1.0, lengthscale=lengthscale_init) 


# SCEGLIERE GLI INDUCING POINTS PER LA SPARSE REGRESSION

def select_inducing_points(X_train, kernel, rho=0.72):
    """
    Implementazione dell'algoritmo OIPS per selezionare automaticamente gli inducing points.
    """
    Z = np.array([X_train[0, :]])  # Inizializza con il primo punto come array NumPy
    for i in range(1, X_train.shape[0]):
        x = X_train[i, :]
        K_xZ = np.array([kernel.K(x.reshape(1, -1), z.reshape(1, -1))[0, 0] for z in Z])
        if np.max(K_xZ) < rho:
            Z = np.vstack([Z, x])  
    return Z

def select_inducing_points_kmeans(X_train, n_clusters):
    """
    Seleziona gli inducing points usando l'algoritmo k-means++.
    
    Parameters:
    X_train (numpy.ndarray): Il dataset di addestramento.
    n_clusters (int): Il numero di inducing points (o centri dei cluster).
    
    Returns:
    numpy.ndarray: I punti inducing selezionati (i centri dei cluster).
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(X_train)
    
    # I centri dei cluster sono i nostri inducing points
    inducing_points = kmeans.cluster_centers_
    
    return inducing_points

#Z = select_inducing_points(X_train_new, kernel, rho=0.72)
Z = select_inducing_points_kmeans(X_train_new, n_clusters=300)


train_or_test = 0 # 1 for train, 0 for test


if train_or_test == 1:
    
    # ALLENARE IL MODELLO SRGP


    # Creazione del modello di regressione GP sparso
    m = gpy.models.SparseGPRegression(X_train_new, log_like_train_norm, kernel=kernel, Z=Z)
    print("Modello iniziale:")
    print(m)
    
    start_time = timeit.default_timer()
    
    # **Ottimizzazione 1: fissiamo gli inducing points e ottimizziamo i parametri del kernel**
    m.inducing_inputs.fix()
    m.optimize('bfgs')
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
    #print(f"Y_var: {Y_var}")
    Y_pred_rescaled = Y_pred * np.std(log_like_train) + np.mean(log_like_train)
    #Y_pred_rescaled = Y_pred * (np.max(log_like_train) - np.min(log_like_train)) + np.min(log_like_train)
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




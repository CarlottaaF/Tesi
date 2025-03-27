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
num_groups = 3

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
log_like_train_r = log_like_train_r.reshape(-1, 1)
log_like_test_r = reshape_loglikelihood(log_like_test, how_many_per_group_test, num_groups)
log_like_test_r = log_like_test_r.reshape(-1, 1)

# Normalize the log-likelihood
# log_like_train_norm = (log_like_train_r - np.mean(log_like_train_r)) / np.std(log_like_train_r)

# oppure
log_like_train_norm = (log_like_train_r - np.min(log_like_train_r)) / (np.max(log_like_train_r) - np.min(log_like_train_r))

# Reshape X
X_train_r = X_train.reshape(-1, X_train.shape[2])
X_test_r = X_test.reshape(-1, X_test.shape[2])

# Reshape Y_exp
Y_exp_train_r = np.tile(Y_exp_train, (num_groups, 1))
Y_exp_test_r = np.tile(Y_exp_test, (num_groups, 1))


#### BUILD THE GP ####

Input_train = np.hstack((X_train_r, Y_exp_train_r))


# Stima del lengthscale prima della selezione degli inducing points
pairwise_dists = pdist(Input_train, metric='euclidean')
lengthscale_init = np.median(pairwise_dists)

# Kernel RBF
kernel = gpy.kern.RBF(input_dim=n_parameters+n_datapoints, variance=1.0, lengthscale=lengthscale_init)


# Inducing points selection
def oips_select_inducing_points(X_train, kernel, rho=0.75):

    Z = [X_train[0]]  # Initialize with the first point
    
    for i in range(1, X_train.shape[0]):
        x = X_train[i].reshape(1, -1)
        Z_array = np.array(Z)
        K_xZ = kernel.K(x, Z_array).flatten()
        if np.max(K_xZ) < rho:
            Z.append(X_train[i])
    
    return np.array(Z)


def select_inducing_points_kmeans(X_train, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(X_train)
    
    # I centri dei cluster sono i nostri inducing points
    inducing_points = kmeans.cluster_centers_
    
    return inducing_points

#Z = oips_select_inducing_points(Input_train, kernel, rho=0.80)
#print(Z.shape)
#Z = select_inducing_points_kmeans(X_train_new, n_clusters=300) --> da sistemare

#selezione casuale degli inducing points
#Z = Input_train[np.random.choice(how_many_per_group_train, 1000, replace=False), :] --> da sistemare

train_or_test = 0 # 1 for train, 0 for test


if train_or_test == 1:
    
    # ALLENARE IL MODELLO SRGP
    
    start_time = timeit.default_timer()

    m_tot = gpy.models.GPRegression(Input_train,log_like_train_norm,kernel)
    print("Modello totale:")
    print(m_tot)
    m_tot.optimize('bfgs')
    print(m_tot)

    # Creazione del modello di regressione GP sparso
    # m = gpy.models.SparseGPRegression(Input_train, log_like_train_norm, kernel=kernel, Z=Z)
    # print("Modello iniziale:")
    # print(m)
    
    
    
    # **Ottimizzazione 1: fissiamo gli inducing points e ottimizziamo i parametri del kernel**

    # m.inducing_inputs.fix()
    # m.optimize('bfgs')
    # print(m)
    
    # **Ottimizzazione 2: sblocchiamo gli inducing points e ottimizziamo tutti i parametri**
    # m.Z.unconstrain()
    # m.optimize('bfgs')
    # print(m)

    elapsed_time = timeit.default_timer() - start_time
    print(f"Tempo di esecuzione: {elapsed_time}")

    # SALVARE IL MODELLO

    with open(save_path, "wb") as f:
        pickle.dump(m_tot, f)  #  Salva l'intero modello
else:

    # CARICARE IL MODELLO
    with open(save_path, "rb") as f:
        m_load = pickle.load(f)  #  Carica il modello senza dover reimpostare i parametri

    print(m_load)

    # FARE PREDIZIONE SUL TEST_SET
    
    Input_test = np.hstack((X_test_r, Y_exp_test_r))
    Y_pred, Y_var = m_load.predict(Input_test) 
    #print(f"Y_var: {Y_var}")
    #Y_pred_rescaled = Y_pred * np.std(log_like_train_r) + np.mean(log_like_train_r)
    Y_pred_rescaled = Y_pred * (np.max(log_like_train_r) - np.min(log_like_train_r)) + np.min(log_like_train_r)
    #print(Y_pred_rescaled)

    # Errore assoluto medio
    mae = np.mean(np.abs(Y_pred_rescaled - log_like_test_r))
    print(f"Errore Assoluto Medio (MAE): {mae}")
    
    # Plot usando hexbin
    plt.figure(figsize=(6.5, 6.5), dpi=100)
    hb = plt.hexbin(log_like_test_r, Y_pred_rescaled, gridsize=50, cmap='plasma', mincnt=1)
    cb = plt.colorbar(hb, label='Frequency')
    
    # Linea guida y=x
    lims = [min(log_like_test_r.min(), Y_pred_rescaled.min()), 
            max(log_like_test_r.max(), Y_pred_rescaled.max())]
    plt.plot(lims, lims, 'gold', linewidth=1.5, zorder=-1, label='y=x')
    
    # Etichette e legenda
    plt.xlabel('True Value [%]')
    plt.ylabel('Prediction [%]')
    plt.title('Hexbin Density Plot')
    plt.legend(loc='upper left') 

    plt.savefig('/data_generation_cartella/images/parity_plot_SRGP.png', format='png', dpi=300, bbox_inches='tight')




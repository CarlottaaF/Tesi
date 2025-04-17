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
from scipy import stats
from scipy.stats import norm
from sklearn.decomposition import PCA
from joblib import dump

save_path = "/data_generation_cartella/GPy_models/sparse_gp_05.pkl"

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


# 1. Prima applica PCA sulle osservazioni pulite
# --------------------------------------------------
# Analisi della varianza spiegata per determinare il numero di componenti
pca = PCA().fit(y_train)

# plt.figure(figsize=(10,6))
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Numero di componenti principali')
# plt.ylabel('Varianza cumulativa spiegata')
# plt.title('Scelta del numero di componenti principali')
# plt.grid(True)
# plt.savefig('/data_generation_cartella/images/pca.png', format='png')

# Scegli il numero di componenti
n_components = 16

# Applica PCA con il numero di componenti selezionato
pca = PCA(n_components=n_components)
y_train_pca = pca.fit_transform(y_train)
y_test_pca = pca.transform(y_test)

dump(pca, '/data_generation_cartella/GPy_models/pca_model.joblib')  # Salva il modello PCA

# Creo le istanze di dati e parametri
def create_data_groups(X, Y, Y_pca, total_instances, how_many_per_group, num_groups, n_datapoints, n_parameters, n_components):

    total_indices = np.arange(total_instances)
    
    # Seleziona gli indici per X_exp e Y_exp (istanze da sporcare)
    indices_exp = np.random.choice(total_indices, size=how_many_per_group, replace=False)
    remaining_indices = np.setdiff1d(total_indices, indices_exp)
    
    # Array per memorizzare i gruppi
    Y_groups = np.empty((num_groups, how_many_per_group, n_datapoints))
    Y_groups_pca = np.empty((num_groups, how_many_per_group, n_components))
    X_groups = np.empty((num_groups, how_many_per_group, n_parameters))
    
    for i in range(num_groups):
        selected_indices = np.random.choice(remaining_indices, size=how_many_per_group, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, selected_indices)
        
        Y_groups[i] = Y[selected_indices]
        Y_groups_pca[i] = Y_pca[selected_indices]
        X_groups[i] = X[selected_indices]
    
    Y_exp = Y[indices_exp]
    Y_exp_pca = Y_pca[indices_exp]
    X_exp = X[indices_exp]
    
    return X_exp, Y_exp, Y_exp_pca, X_groups, Y_groups, Y_groups_pca

how_many_per_group_train = 900
how_many_per_group_test = 100
num_groups = 10

X_exp_train, Y_exp_train, Y_exp_train_pca, X_train_groups, Y_train, Y_train_pca = create_data_groups(X_train, y_train, y_train_pca, tot_samples_train, how_many_per_group_train, num_groups, n_datapoints, n_parameters, n_components)
X_exp_test, Y_exp_test, Y_exp_test_pca, X_test_groups, Y_test, Y_test_pca = create_data_groups(X_test, y_test, y_test_pca, tot_samples_test, how_many_per_group_test, num_groups, n_datapoints, n_parameters, n_components)

# Add noise to the data
noise = 0.01
Y_exp_train_pca = Y_exp_train_pca + np.random.normal(loc=0.0, scale=noise, size=Y_exp_train_pca.shape)  
Y_exp_test_pca = Y_exp_test_pca + np.random.normal(loc=0.0, scale=noise, size=Y_exp_test_pca.shape)  

Y_exp_train = Y_exp_train + np.random.normal(loc=0.0, scale=noise, size=Y_exp_train.shape)  
Y_exp_test = Y_exp_test + np.random.normal(loc=0.0, scale=noise, size=Y_exp_test.shape) 


# Compute the log-likelihood
def compute_loglikelihood(Y_exp, Y, how_many_per_group, n_groups, sigma, n_datapoints):

    log_lik = np.zeros((how_many_per_group, n_groups)) 
    
    for i1 in range(how_many_per_group):
        for i2 in range(n_groups):
                somma = 0
                for i3 in range(n_datapoints):
                    somma += -(((Y_exp[i1,i3]-Y[i2,i1,i3]) ** 2) / (2. * (10*sigma) ** 2)) #+np.log(1. / (np.sqrt(2. * np.pi) * (sigma*10)))
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
# print(np.mean(log_like_train_r))
# print(np.std(log_like_train_r))
log_like_train_norm = (log_like_train_r - np.min(log_like_train_r)) / (np.max(log_like_train_r) - np.min(log_like_train_r))


# Reshape X
X_train_r = X_train_groups.reshape(-1, X_train_groups.shape[2])
X_test_r = X_test_groups.reshape(-1, X_test_groups.shape[2])

# Reshape Y_exp
Y_exp_train_pca_r = np.tile(Y_exp_train_pca, (num_groups, 1))
Y_exp_test_pca_r = np.tile(Y_exp_test_pca, (num_groups, 1))


#### BUILD THE GP ####

Input_train = np.hstack((Y_exp_train_pca_r, X_train_r))




#kernel = gpy.kern.RBF(input_dim=n_components + n_parameters,name='kernel', ARD = False)

# Kernel per i dati (25 features): cattura correlazioni spaziali
kernel_data = gpy.kern.RBF(input_dim=n_components, active_dims=range(n_components), ARD = True, name='data_kernel')

# Kernel per i parametri (16 features): cattura dipendenze tra parametri KLE
kernel_params = gpy.kern.RBF(input_dim=n_parameters, active_dims=range(n_components, n_parameters+n_components), ARD = True, name='param_kernel')

# Kernel moltiplicativo (interazione tra dati e parametri)
kernel = kernel_data * kernel_params

def select_inducing_points_kmeans(X_train, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, max_iter=300, random_state=42)
    kmeans.fit(X_train)
    inducing_points = kmeans.cluster_centers_
    
    return inducing_points



Z = select_inducing_points_kmeans(Input_train, 400) 
#Z = Input_train[np.random.choice(how_many_per_group_train, 400, replace=False), :]

train_or_test = 1 # 1 for train, 0 for test


if train_or_test == 1:
    
    # ALLENARE IL MODELLO SRGP
    
    start_time = timeit.default_timer()


    # m = gpy.models.GPRegression(Input_train,log_like_train_norm,kernel)
    # print("Modello totale:")
    # print(m)
    # m.optimize('bfgs')
    # print(m)

    m = gpy.models.SparseGPRegression(X=Input_train, Y=log_like_train_norm, kernel=kernel, Z=Z, name='Sparse_GP')
    print("Modello SRGP:")
    print(m)

    m.inducing_inputs.fix()
    m.optimize('bfgs')
    print(m)

    m.inducing_inputs.unfix()  # Sblocca gli inducing points
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

    #print(m_load.kern.parts[0].lengthscale) 
    #print(m_load.kern.parts[1].lengthscale)
    #print(m_load.kern.lengthscale)

    print(m_load.log_likelihood())

    # FARE PREDIZIONE SUL TEST_SET
    
    Input_test = np.hstack((Y_exp_test_pca_r,X_test_r ))
    Y_pred, Y_var = m_load.predict(Input_test) 
    #print(Y_var)
    #print(m_load.rbf.lengthscale)
    #print(f"Y_var: {Y_var}")
    #Y_pred_rescaled = Y_pred * np.std(log_like_train_r) + np.mean(log_like_train_r)
    Y_pred_rescaled = Y_pred * (np.max(log_like_train_r) - np.min(log_like_train_r)) + np.min(log_like_train_r)
    print(np.min(log_like_train_r))
    print(np.max(log_like_train_r))
    #print(Y_pred_rescaled)


    
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

    plt.savefig('/data_generation_cartella/images/parity_plot_sparse_GP_05.png', format='png', dpi=300, bbox_inches='tight')

    residuals = log_like_test_r - Y_pred_rescaled
    
    #print(residuals)

    plt.figure(figsize=(6.5, 6.5), dpi=100)
    plt.scatter(Y_pred_rescaled, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Value")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residual Plot")
    plt.savefig('/data_generation_cartella/images/Residual_plot_sparse_GP_05.png', format='png')
    
    plt.figure(figsize=(6.5, 6.5), dpi=100)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.savefig('/data_generation_cartella/images/Distribution_plot_sparse_GP_05.png', format='png')


    # def compute_coverage(y_true, y_pred_mean, y_pred_std, alpha=0.95):
    #     z = norm.ppf(0.5 + alpha / 2)  # per 95%, z ≈ 1.96
    #     lower = y_pred_mean - z * y_pred_std
    #     upper = y_pred_mean + z * y_pred_std
    #     covered = (y_true >= lower) & (y_true <= upper)
    #     return np.mean(covered)
    
    # print("Coverage:")
    # Y_var_originale = (Y_var * (np.max(log_like_train_r) - np.min(log_like_train_r))**2)
    # coverage = compute_coverage(log_like_test_r, Y_pred_rescaled, np.sqrt(Y_var_originale), alpha=0.95)
    # print(coverage)


  


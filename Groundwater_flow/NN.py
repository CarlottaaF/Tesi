from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from scipy.stats import qmc
import sys
from sklearn.decomposition import PCA

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

# PCA
pca = PCA().fit(y_train)

# Scegli il numero di componenti
n_components = 16

# Applica PCA con il numero di componenti selezionato
pca = PCA(n_components=n_components)
y_train_pca = pca.fit_transform(y_train)
y_test_pca = pca.transform(y_test)

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
num_groups = 50

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
                    somma += -(((Y_exp[i1,i3]-Y[i2,i1,i3]) ** 2) / (2. * (10*sigma) ** 2)) 
                log_lik[i1, i2] = somma 
    
    return log_lik

log_like_train = compute_loglikelihood(Y_exp_train, Y_train, how_many_per_group_train, num_groups, noise, n_datapoints)
log_like_test = compute_loglikelihood(Y_exp_test, Y_test, how_many_per_group_test, num_groups, noise, n_datapoints)


# Reshape the log-likelihood
def reshape_loglikelihood(log_like, how_many_per_group, n_groups):

    likelihood_r = np.reshape(log_like, (how_many_per_group* n_groups,),'F') # dati riorganizzati per colonne 
    return likelihood_r

log_like_train_r = reshape_loglikelihood(log_like_train, how_many_per_group_train, num_groups)
#log_like_train_r = log_like_train_r.reshape(-1, 1)
log_like_test_r = reshape_loglikelihood(log_like_test, how_many_per_group_test, num_groups)
#log_like_test_r = log_like_test_r.reshape(-1, 1)


# Normalize the log-likelihood
log_like_train_norm = (log_like_train_r - np.min(log_like_train_r)) / (np.max(log_like_train_r) - np.min(log_like_train_r))
print(np.min(log_like_train_r))
print(np.max(log_like_train_r))


# Reshape X
X_train_r = X_train_groups.reshape(-1, X_train_groups.shape[2])
X_test_r = X_test_groups.reshape(-1, X_test_groups.shape[2])

# Reshape Y_exp
Y_exp_train_pca_r = np.tile(Y_exp_train_pca, (num_groups, 1))
Y_exp_test_pca_r = np.tile(Y_exp_test_pca, (num_groups, 1))


# Specify if you want to train the net or use it to make predictions (0-predict ; 1-train)
predict_or_train = 0

# # Hyperparameters
validation_split = 0.20
batch_size = 32
n_epochs = 150
early_stop_epochs=20
initial_lr = 1e-3
decay_length = 0.8
ratio_to_stop = 0.05
rate_drop = 0.05


if predict_or_train:

        input_series  = layers.Input(shape=(n_components,), name='Recordings_inputs')
        input_params  = layers.Input(shape=(n_parameters,), name='Parameters_inputs')

        y = layers.Dense(128, activation='relu')(input_series)
        
        merged = layers.Concatenate()([y, input_params])
        
        z = layers.Dense(128, activation='relu')(merged)

        output = layers.Dense(1, activation='linear')(z)

        Regressor = keras.models.Model([input_series, input_params], output)
        Regressor.compile(optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr, decay_steps=int(decay_length*n_epochs*how_many_per_group_train**2*(1-validation_split)/batch_size), alpha=ratio_to_stop)),
                            loss='mse',
                            metrics=['mae'])

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_epochs, restore_best_weights=True)
        
        # Addestramento
        history = Regressor.fit([Y_exp_train_pca_r, X_train_r],
                                log_like_train_norm,
                                epochs=n_epochs,  
                                batch_size=batch_size,
                                validation_split=0.2,
                                verbose=2,
                                callbacks=[early_stop])

            
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_pickle("/data_generation_cartella/NN/hist_02.pkl")
        Regressor.save("/data_generation_cartella/NN/model_02.keras")

else:        
        Regressor = keras.models.load_model("/data_generation_cartella/NN/model_02.keras")
        
        with open("/data_generation_cartella/NN/hist_02.pkl", 'rb') as file:
            hist = pickle.load(file)
           
        
        #model test
        
        predictions = Regressor.predict([Y_exp_test_pca_r, X_test_r]).flatten() 
        
        #riscalo le predizioni in modo che non siano più tra zero e uno
        predictions_true = predictions*(np.max(log_like_train_r) - np.min(log_like_train_r)) + np.min(log_like_train_r)
                
        # Plot usando hexbin
        plt.figure(figsize=(6.5, 6.5), dpi=100)
        hb = plt.hexbin(log_like_test_r, predictions_true, gridsize=50, cmap='plasma', mincnt=1)
        cb = plt.colorbar(hb, label='Frequency')
        
        # Linea guida y=x
        lims = [min(log_like_test_r.min(), predictions_true.min()), 
                max(log_like_test_r.max(), predictions_true.max())]
        plt.plot(lims, lims, 'gold', linewidth=1.5, zorder=-1, label='y=x')
        
        # Etichette e legenda
        plt.xlabel('True Value [%]')
        plt.ylabel('Prediction [%]')
        plt.title('Hexbin Density Plot')
        plt.legend(loc='upper left') 

        plt.savefig('/data_generation_cartella/NN/parity_plot_NN_02.png', format='png', dpi=300, bbox_inches='tight')
        
        residuals = log_like_test_r - predictions_true

        plt.figure(figsize=(6.5, 6.5), dpi=100)
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Errors")
        plt.savefig('/data_generation_cartella/NN/Distribution_plot_NN_02.png', format='png')

        plt.figure(figsize=(6.5, 6.5), dpi=100)
        plt.scatter(predictions_true, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Value")
        plt.ylabel("Residuals (True - Predicted)")
        plt.title("Residual Plot")
        plt.savefig('/data_generation_cartella/NN/Residual_plot_NN_02.png', format='png')
                

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:21:31 2024

@author: Ruggi
"""
#%%

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
from sklearn.model_selection import train_test_split


# Imposta la directory corrente come percorso di lavoro
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(current_dir)

import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%
esempio = 'L_FRAME_MF_HF_RETE' # Specify the example you are dealing with
ID_test               = '1_0_MCMC_3'
work_ID         = 'Save\\EPSR\\'
path            = 'C:\\Users\\Ruggi\\Carlotta\\' + esempio + '\\Dati\\'
path_data_test   = path + 'istantest_' + ID_test
work_path       = path + work_ID
ID_HF      = 'MF_HF_RETE_1_0'
HF_NN_path    = path + ID_HF
ID_LF           = '1_0'
LF_NN_path      = 'C:\\Users\\Ruggi\\Carlotta\\L_FRAME_MF_LF_RETE\\Dati\\MF_LF_RETE_'+ID_LF
ID_basis        = '1_0'
LF_basis_path      = 'C:\\Users\\Ruggi\\Carlotta\\L_FRAME_MF_LF_RETE\\Dati\\istantrain_'+ID_basis


save_ID         = 'Regressor_1\\'
path_save       = path + save_ID
restore_ID= 'Regressor_1\\'
path_restore = path + restore_ID

#%%

### LOAD HF SIGNALS ####

N_ist = 1000
n_channels = 8
N_entries = 200

file_paths = [
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_1.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_2.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_3.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_4.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_5.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_6.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_7.csv',
    'C:\\Users\\Ruggi\\Carlotta\\istantrain_1_0\\U_concat_8.csv'
]

Y_HF = utils.load_HF_signals(file_paths, N_ist, n_channels, N_entries)

#%%

### SAMPLE HF INPUT ####

n_ist_par = 10 # numero di diverse istanze di parametri generate da LHS
X_HF = np.zeros((n_ist_par,N_ist,4)) #struttura che contiene, per ogni istanza di parametri, le mille istanze di ampiezza,frequenza e coordinate danno

# for i1 in range(n_ist_par):
#     sampler = qmc.LatinHypercube(d=3) #ho 3 input: frequenza, ampiezza e coord danno
#     sample = sampler.random(n=1000) #voglio generare 1000 valori per i miei 3 input
#     X_HF[i1,:,0] = sample[:,0]
#     X_HF[i1,:,1] = sample[:,1]
#     for i2 in range(1000):
#         if sample[i2, 2] <= 0.5:  #  se il danno è nel primo braccio della struttura
#             X_HF[i1,i2, 2] = sample[i2, 2] * 2 #coordinata x danno
#             X_HF[i1,i2, 3] = 0 #coordinata y danno
#         else:
#             X_HF[i1,i2, 2] = 1
#             X_HF[i1,i2, 3] = (sample[i2, 2] - 0.5) * 2
            
for i1 in range(n_ist_par):
    sampler = qmc.LatinHypercube(d=1) #ho 1 input: coord danno
    sample = sampler.random(n=1000) #voglio generare 1000 valori per il mio input

    for i2 in range(1000):
        if sample[i2, 0] <= 0.5:  #  se il danno è nel primo braccio della struttura
            X_HF[i1,i2, 2] = sample[i2, 0] * 2 #coordinata x danno
            X_HF[i1,i2, 3] = 0 #coordinata y danno
        else:
            X_HF[i1,i2, 2] = 1
            X_HF[i1,i2, 3] = (sample[i2, 0] - 0.5) * 2
            
for i1 in range(n_ist_par):
    X_HF[i1,:,0] = utils.input_true()[:,0]
    X_HF[i1,:,1] = utils.input_true()[:,1]
    

#%%

### SPLIT IN TRAINING E TEST SET ###

# Split del dataset: 90% per il training e 10% per il test
train_size = 0.9  # Percentuale per il training set
test_size = 1 - train_size  # Percentuale per il test set

indices = np.arange(N_ist)  # Array degli indici delle istanze
train_indices, test_indices = train_test_split(indices, test_size=test_size)

# Creazione dei set di training e test
Y_HF_train = Y_HF[train_indices]
Y_HF_test = Y_HF[test_indices]

N_ist_train = 900
N_ist_test = 100

#%%

### SPORCARE Y_HF CON RUMORE ####

SNR = 100

Y_exp_train = utils.add_noise_to_dataset(Y_HF_train, SNR)
Y_exp_test = utils.add_noise_to_dataset(Y_HF_test, SNR)

#%%

### NORMALIZATION OF THE DATA ###

Y_exp_train_norm,media_train,std_train = utils.normalize_dataset(Y_exp_train)
Y_exp_test_norm,_,_ = utils.normalize_dataset(Y_exp_test,media_train,std_train)

#%%

### SETUP OF THE SURROGATE MODEL ###

basis = np.load(LF_basis_path+'\\basis.pkl',allow_pickle=True) #import base PCA per modello LF
N_basis = np.size(basis,1) #numero componenti principali
weights = np.linspace(1,0.2,N_basis) #pesi di scaling per le componenti principali

###########

LF_net = keras.models.load_model(LF_NN_path + '\\LF_model', compile=False) #import modello LF
LF_mean = np.load(LF_NN_path+'\\mean_LF_POD.npy') #import mean normalizzazione output LF
LF_std = np.load(LF_NN_path+'\\std_LF_POD.npy') #import stdv normalizzazione output LF
LF_signals_means = np.load(LF_NN_path+'\\LF_signals_means.npy') #import mean normalizzazione segnali LF
LF_signals_stds = np.load(LF_NN_path+'\\LF_signals_stds.npy') #import stdv normalizzazione segnali LF
HF_net_trained = keras.models.load_model(HF_NN_path + '\\HF_model', compile=False) #import modello HF
HF_signals_means = np.load(HF_NN_path+'\\HF_signals_means.npy') #import mean normalizzazione segnali HF
HF_signals_stds = np.load(HF_NN_path+'\\HF_signals_stds.npy') #import stdv normalizzazione segnali HF

#costruisco modello HF
HF_neurons_LSTM_1 = 16
HF_neurons_LSTM_2 = 16
HF_neurons_LSTM_3 = 32

HF_input = layers.Input(shape=(N_entries,5+n_channels), name='input_hf')
x = layers.LSTM(units=HF_neurons_LSTM_1, return_sequences=True, name='Reccurrent_1')(HF_input)                  
x = layers.LSTM(units=HF_neurons_LSTM_2, return_sequences=True, name='Reccurrent_2')(x)                         
x = layers.LSTM(units=HF_neurons_LSTM_3, return_sequences=True, name='Reccurrent_3')(x)                        
x = layers.LSTM(units=n_channels, return_sequences=True, name='Reccurrent_4')(x)                      
HF_output = layers.Dense(units=n_channels, activation=None, name='Linear_mapping')(x)   
HF_net_to_pred = keras.Model(inputs=HF_input, outputs=HF_output, name="HF_model_prediction")

for i1 in range(len(HF_net_trained.layers)): #prendo pesi da modello caricato e li attacco al modello costruito
    HF_net_to_pred.layers[i1].set_weights(HF_net_trained.layers[i1].get_weights())
    
#%%

### APPLY THE SURROGATE MODEL ###

signal_resolution = 0.005 #time step segnale

# Applicazione sul training set
Y_train,Input_HF_train = utils.apply_surrogate_model(X_HF, train_indices, n_ist_par, N_entries, n_channels, 
                                signal_resolution, LF_net, HF_net_to_pred, 
                                LF_mean, LF_std, weights, basis, LF_signals_means, LF_signals_stds)

# Applicazione sul test set
Y_test,Input_HF_test = utils.apply_surrogate_model(X_HF, test_indices, n_ist_par, N_entries, n_channels, 
                               signal_resolution, LF_net, HF_net_to_pred, 
                               LF_mean, LF_std, weights, basis, LF_signals_means, LF_signals_stds)

#%%

### COMPUTE THE LOG-LIKELIHOOD ###

removed_ratio = 0.2 #ratio of N_entries to be removed in the computation of the likelihood
limit=int(N_entries*removed_ratio) #numero di time step da rimuovere

# Calcolo della log-likelihood sul training set
likelihood_train = utils.compute_likelihood(Y_exp_train_norm, Y_train, N_ist_train, n_ist_par, n_channels, N_entries, removed_ratio, limit)

# Calcolo della log-likelihood sul test set
likelihood_test = utils.compute_likelihood(Y_exp_test_norm, Y_test, N_ist_test, n_ist_par, n_channels, N_entries, removed_ratio, limit)

#%%

### RESHAPE STRUTTURE PER AUTOENCODER ###

likelihood_r_train = utils.likelihood_reshaped(likelihood_train, N_ist_train, n_ist_par)

#normalizzo likelihood
lik_min_train = np.min(likelihood_r_train) #-38.14900633995325
lik_max_train = np.max(likelihood_r_train)
delta_max_train = lik_max_train-lik_min_train #48.25939812179713
likelihood_r_norm_train = (likelihood_r_train - lik_min_train)/delta_max_train

Input_HF_r_train = utils.reshape_input_HF(Input_HF_train, N_ist_train, n_ist_par)

Y_exp_r_train = utils.reshape_Y_exp(Y_exp_train_norm, N_ist_train, n_ist_par, n_channels, N_entries)

#%%
 
 
### BUILD THE AUTOENCODER ###

#ottimizzazione bayesiana
# Specify if you want to train the net or use it to make predictions (0-predict ; 1-train)
predict_or_train = 0

# # Hyperparameters
validation_split = 0.20
batch_size = 32
n_epochs = 150
early_stop_epochs=15
initial_lr = 0.00196
decay_length = 0.86
ratio_to_stop = 0.05
k_reg = 1.76638e-07
rate_drop = 0.05

if predict_or_train:

        try: 
             os.mkdir(path_save) 
        except: 
             print('ERRORE: La cartella esiste giÃ ')
             
        input_series  = layers.Input(shape=(N_entries, n_channels), name='Convolutional_inputs')
        input_params  = layers.Input(shape=(4,), name='Parameters_inputs')
             
        
        # Blocchi convoluzionali
        x = layers.Conv1D(filters=64, kernel_size=13, activation='relu', kernel_regularizer=regularizers.l2(k_reg),padding='same')(input_series)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=13, activation='relu', kernel_regularizer=regularizers.l2(k_reg), padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=32, kernel_size=25, activation='relu', kernel_regularizer=regularizers.l2(k_reg), padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Flatten()(x)
        
        # Layer completamente connessi
        x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)
        x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)
        x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)


        
        x = layers.Concatenate()([x, input_params])
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(k_reg))(x)

     
        # Output layer per la regressione
        output = layers.Dense(1, activation='linear')(x)
  
        
        # Definizione del modello
        Regressor = keras.models.Model([input_series, input_params], output, name='Regressor_Model')
        Regressor.summary()
        
        # Compilazione

        Regressor.compile(optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.0014, decay_steps=int(0.74*n_epochs*N_ist**2*(1-validation_split)/batch_size), alpha=ratio_to_stop)),
                            loss='mse',
                            metrics=['mae'])

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_epochs, restore_best_weights=True)
        
        # Addestramento
        history = Regressor.fit([Y_exp_r_train, Input_HF_r_train[:, 0:4]],
                                likelihood_r_norm_train,
                                epochs=n_epochs,  
                                batch_size=batch_size,
                                validation_split=0.2,
                                verbose=2,
                                callbacks=[early_stop])

            
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_pickle(path_save + 'hist.pkl')
        Regressor.save(path_save + 'model')

else:        
        Regressor = keras.models.load_model(path_restore + 'model')
        
        with open(path_restore + 'hist.pkl', 'rb') as file:
            hist = pickle.load(file)
           
        
        #model test

        Input_HF_r_test = utils.reshape_input_HF(Input_HF_test, N_ist_test, n_ist_par)
        Y_exp_r_test = utils.reshape_Y_exp(Y_exp_test_norm, N_ist_test, n_ist_par, n_channels, N_entries)
        
        predictions = Regressor.predict([Y_exp_r_test, Input_HF_r_test[:,0:4]]).flatten() 
        
        #riscalo le predizioni in modo che non siano più tra zero e uno
        predictions_true = predictions*delta_max_train + lik_min_train
                
        likelihood_r_test=utils.likelihood_reshaped(likelihood_test, N_ist_test, n_ist_par)
        
        
        # Plot usando hexbin
        plt.figure(figsize=(6.5, 6.5), dpi=100)
        hb = plt.hexbin(likelihood_r_test, predictions_true, gridsize=50, cmap='plasma', mincnt=1)
        cb = plt.colorbar(hb, label='Frequency')
        
        # Linea guida y=x
        lims = [min(likelihood_r_test.min(), predictions_true.min()), 
                max(likelihood_r_test.max(), predictions_true.max())]
        plt.plot(lims, lims, 'gold', linewidth=1.5, zorder=-1, label='y=x')
        
        # Etichette e legenda
        plt.xlabel('True Value [%]')
        plt.ylabel('Prediction [%]')
        plt.title('Hexbin Density Plot')
        plt.legend(loc='upper left') 
        
        plt.show()


#%%
# Predict con i parametri veri

X1_HF = utils.input_true()
X1_HF_test = X1_HF[test_indices]
Input_HF_true = np.zeros((n_ist_par, N_ist_test, N_entries, 5 + n_channels))
Input_HF_true[:,:,:,4:13] = Input_HF_test[:,:,:,4:13]

for i1 in range(n_ist_par):
  for i3 in range(N_ist_test):
    Input_HF_true[i1, i3, :, 0] = X1_HF_test[i3, 0]  # Frequenza
    Input_HF_true[i1, i3, :, 1] = X1_HF_test[i3, 1]  # Ampiezza
    Input_HF_true[i1, i3, :, 2] = X1_HF_test[i3, 2]  # Coord_x danno
    Input_HF_true[i1, i3, :, 3] = X1_HF_test[i3, 3]  # Coord_y danno
    
    
Input_HF_true_r = utils.reshape_input_HF(Input_HF_true,N_ist_test,n_ist_par)

predictions_input_true = Regressor.predict([Y_exp_r_test, Input_HF_true_r[:,0:4]]).flatten()
predictions_true_input_true = predictions_input_true*delta_max_train+lik_min_train

Y_true = np.zeros((n_ist_par, N_ist_test, n_channels, N_entries)) 
for i1 in range(n_ist_par):
    Y_i = HF_net_to_pred.predict(Input_HF_true[i1, :, :, :], verbose=0)
    Y_true[i1] = np.transpose(Y_i, (0, 2, 1))

likelihood_true = utils.compute_likelihood(Y_exp_test_norm, Y_true, N_ist_test, n_ist_par, n_channels, N_entries, removed_ratio, limit)
likelihood_r_true=utils.likelihood_reshaped(likelihood_true, N_ist_test, n_ist_par)

# Plot usando hexbin
plt.figure(figsize=(6.5, 6.5), dpi=100)
hb = plt.hexbin(likelihood_r_true, predictions_true_input_true, gridsize=50, cmap='plasma', mincnt=1)
cb = plt.colorbar(hb, label='Frequency')

# Linea guida y=x
lims = [min(likelihood_r_true.min(), predictions_true_input_true.min()), 
        max(likelihood_r_true.max(), predictions_true_input_true.max())]
plt.plot(lims, lims, 'gold', linewidth=1.5, zorder=-1, label='y=x')

# Etichette e legenda
plt.xlabel('True Value [%]')
plt.ylabel('Prediction [%]')
plt.title('Hexbin Density Plot considering the true input')
plt.legend(loc='upper left') 

plt.show()


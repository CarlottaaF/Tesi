# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:25:51 2025

@author: Ruggi
"""

#%%
import logging
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
import optuna

# Imposta la directory corrente come percorso di lavoro
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(current_dir)

import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#%%
esempio = 'L_FRAME_MF_HF_RETE' # Specify the example you are dealing with
ID_test               = '1_0_MCMC_3'
work_ID         = 'Save/EPSR/'
path            = './Carlotta/' + esempio + '/Dati/'
path_data_test   = path + 'istantest_' + ID_test
work_path       = path + work_ID
ID_HF      = 'MF_HF_RETE_1_0'
HF_NN_path    = path + ID_HF
ID_LF           = '1_0'
LF_NN_path      = './Carlotta/L_FRAME_MF_LF_RETE/Dati/MF_LF_RETE_'+ID_LF
ID_basis        = '1_0'
LF_basis_path      = './Carlotta/L_FRAME_MF_LF_RETE/Dati/istantrain_'+ID_basis


save_ID         = 'Regressor_1/'
path_save       = path + save_ID
restore_ID= 'Regressor_1/'
path_restore = path + restore_ID

#%%

### LOAD HF SIGNALS ####

N_ist = 1000
n_channels = 8
N_entries = 200

file_paths = [
    './Carlotta/istantrain_1_0/U_concat_1.csv',
    './Carlotta/istantrain_1_0/U_concat_2.csv',
    './Carlotta/istantrain_1_0/U_concat_3.csv',
    './Carlotta/istantrain_1_0/U_concat_4.csv',
    './Carlotta/istantrain_1_0/U_concat_5.csv',
    './Carlotta/istantrain_1_0/U_concat_6.csv',
    './Carlotta/istantrain_1_0/U_concat_7.csv',
    './Carlotta/istantrain_1_0/U_concat_8.csv'
]

Y_HF = utils.load_HF_signals(file_paths, N_ist, n_channels, N_entries)

#%%

### SAMPLE HF INPUT ####

n_ist_par = 10 # numero di diverse istanze di parametri generate da LHS
X_HF = np.zeros((n_ist_par,N_ist,4)) #struttura che contiene, per ogni istanza di parametri, le mille istanze di ampiezza,frequenza e coordinate danno

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

basis = np.load(LF_basis_path+'/basis.pkl',allow_pickle=True) #import base PCA per modello LF
N_basis = np.size(basis,1) #numero componenti principali
weights = np.linspace(1,0.2,N_basis) #pesi di scaling per le componenti principali

###########

LF_net = keras.models.load_model(LF_NN_path + '/LF_model', compile=False) #import modello LF
LF_mean = np.load(LF_NN_path+'/mean_LF_POD.npy') #import mean normalizzazione output LF
LF_std = np.load(LF_NN_path+'/std_LF_POD.npy') #import stdv normalizzazione output LF
LF_signals_means = np.load(LF_NN_path+'/LF_signals_means.npy') #import mean normalizzazione segnali LF
LF_signals_stds = np.load(LF_NN_path+'/LF_signals_stds.npy') #import stdv normalizzazione segnali LF
HF_net_trained = keras.models.load_model(HF_NN_path + '/HF_model', compile=False) #import modello HF
HF_signals_means = np.load(HF_NN_path+'/HF_signals_means.npy') #import mean normalizzazione segnali HF
HF_signals_stds = np.load(HF_NN_path+'/HF_signals_stds.npy') #import stdv normalizzazione segnali HF

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
lik_min_train = np.min(likelihood_r_train)
lik_max_train = np.max(likelihood_r_train)
delta_max_train = lik_max_train-lik_min_train
likelihood_r_norm_train = (likelihood_r_train - lik_min_train)/delta_max_train

Input_HF_r_train = utils.reshape_input_HF(Input_HF_train, N_ist_train, n_ist_par)

Y_exp_r_train = utils.reshape_Y_exp(Y_exp_train_norm, N_ist_train, n_ist_par, n_channels, N_entries)

#%%
 
# Funzione obiettivo per Optuna
# OTTIMIZZAZIONE BAYESIANA
def objective(trial):
    # Hyperparameters suggested by Optuna
    # massimo 12 iperparametri
    filters_1 = trial.suggest_categorical('filters_1', [32, 64, 128])
    kernel_size_1 = trial.suggest_categorical('kernel_size_1', [7,13,25])
    filters_2 = trial.suggest_categorical('filters_2', [32, 64, 128])
    kernel_size_2 = trial.suggest_categorical('kernel_size_2', [7,13,25])
    filters_3 = trial.suggest_categorical('filters_3', [16, 32, 64])
    kernel_size_3 = trial.suggest_categorical('kernel_size_3', [13,25,50])
    k_reg = trial.suggest_loguniform('k_reg', 1e-9, 1e-6)
    learning_rate = 1e-3
    batch_size = 32
    neurons_1 = trial.suggest_categorical('neurons_1', [8, 16, 32, 64, 128, 256])
    neurons_2 = trial.suggest_categorical('neurons_2', [8, 16, 32, 64, 128, 256])
    n_layers1 = trial.suggest_int('n_layers1', 1, 5) #number of layers before concatenate
    n_layers2 = trial.suggest_int('n_layers2', 1, 5) #number of layers after concatenate
    activation = trial.suggest_categorical('activation', ['tanh', 'selu', 'gelu', 'relu'])
    activation1 = trial.suggest_categorical('activation1', ['tanh', 'selu', 'gelu', 'relu'])
    activation2 = trial.suggest_categorical('activation2', ['tanh', 'selu', 'gelu', 'relu'])
    decay_length = 0.6

    n_epochs = 200  

    # Inputs
    input_series = layers.Input(shape=(N_entries, n_channels), name='Convolutional_inputs')
    input_params = layers.Input(shape=(4,), name='Parameters_inputs')

    # Convolutional blocks
    x = layers.Conv1D(filters=filters_1, kernel_size=kernel_size_1, activation=activation,
                      kernel_regularizer=regularizers.l2(k_reg), padding='same')(input_series)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=filters_2, kernel_size=kernel_size_2, activation=activation,
                      kernel_regularizer=regularizers.l2(k_reg), padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=filters_3, kernel_size=kernel_size_3, activation=activation,
                      kernel_regularizer=regularizers.l2(k_reg), padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)

    # Fully connected layers

    for _ in range(n_layers1):
        x = layers.Dense(neurons_1, activation=activation1, kernel_regularizer=regularizers.l2(k_reg))(x)

    x = layers.Concatenate()([x, input_params])
    
    for _ in range(n_layers2):
        x = layers.Dense(neurons_2, activation=activation2, kernel_regularizer=regularizers.l2(k_reg))(x)


    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Create the model
    model = keras.models.Model([input_series, input_params], output, name='Regressor_Model')
    model.summary()

    # Compile the model
    ratio_to_stop = 0.05
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate, decay_steps=int(decay_length*n_epochs*N_ist**2*(1-0.2)/batch_size), alpha=ratio_to_stop)),
                        loss='mse',
                        metrics=['mae'])

    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Training
    history = model.fit([Y_exp_r_train, Input_HF_r_train[:, 0:4]],
                        likelihood_r_norm_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=[early_stop])

    val_loss = history.history['val_loss'][-1]
    return val_loss  

# 2. Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')  
study.optimize(objective, n_trials=2000, n_jobs=4)

logging.info(f"Best Hyperparameters: {study.best_params}")

# Save the best parameters
best_params = study.best_params

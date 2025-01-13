# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:09:47 2024

@author: Ruggi
"""


#%%
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import tensorflow as tf
import tinyDA as tda
import time

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

restore_ID= 'Regressor_1\\'
path_restore = path + restore_ID

Regressor = keras.models.load_model(path_restore + 'model')

#%%
n_channels     = 8 #numero canali
SNR = 80 #signal to noise ratio
N_entries = 200 #length of single signal
signal_resolution = 0.005 #time step segnale

removed_ratio = 0.2 #ratio of N_entries to be removed in the computation of the likelihood
limit=int(N_entries*removed_ratio) #numero di time step da rimuovere

freq_min = 10 #valore minimo parametro frequenza
freq_delta_max = 50 #delta range rispetto al minimo parametro frequenza
Ampl_min = 1000 #valore minimo parametro ampiezza
Ampl_delta_max = 4000 #delta range rispetto al minimo parametro ampiezza
Coord_x_min = 0.15 #valore minimo parametro danno in x
Coord_x_delta_max = 3.7 #delta range rispetto al minimo parametro ampiezza
Coord_y_min = 0.15 #valore minimo parametro danno in y
Coord_y_delta_max = 3.7 #delta range rispetto al minimo parametro ampiezza

#MCMC parameters
N_obs = 8 # number of observations for each scenario -> necessario fattore di scala nel calcolo della likelihood
thinning = 4 #subsampling ratio to make samples independent
n_chains = 1

n_parameters = 1 #parametro di danneggiamento

#Apply MCMC for a testing instance (for each we have N_obs observations)
which_ist = 1 #tra 0 e 9


#%%

def read_HF_input(path_data): # carico parametri di input HF
    
    #labels danno in pacchetti da 80 (N_ist*N_obs)
    Coord_x_path = path_data + '\\Coord_x.csv'
    Coord_x      = np.genfromtxt(Coord_x_path) #fisso per N_obs
    Coord_y_path = path_data + '\\Coord_y.csv'                                 
    Coord_y      = np.genfromtxt(Coord_y_path) #fisso per N_obs
    
    #normalizzo dati in ingresso
    Coord_x = (Coord_x - Coord_x_min) / Coord_x_delta_max #normalizzato tra 0 e 1
    Coord_y = (Coord_y - Coord_y_min) / Coord_y_delta_max #normalizzato tra 0 e 1
    N_ist = len(Coord_x) #(N_ist*N_obs)
    
    #organizzo dati
    X_HF = np.zeros((N_ist,2))
    X_HF[:,0] = Coord_x
    X_HF[:,1] = Coord_y
    
    #labels usage in pacchetti da 10 (valide per N_obs)
    Frequency_true_path = path_data + '\\Frequency_true.csv'                
    Frequency_true      = np.genfromtxt(Frequency_true_path) #fisso per N_obs
    Amplitude_true_path = path_data + '\\Amplitude_true.csv'                   
    Amplitude_true      = np.genfromtxt(Amplitude_true_path) #fisso per N_obs
    
    #normalizzo dati in ingresso
    Frequency_true = (Frequency_true - freq_min) / freq_delta_max #normalizzato tra 0 e 1
    Amplitude_true = (Amplitude_true - Ampl_min) / Ampl_delta_max #normalizzato tra 0 e 1
    
    #organizzo dati
    X_true = np.zeros((N_ist//N_obs,2))
    X_true[:,0] = Frequency_true
    X_true[:,1] = Amplitude_true
        
    return X_HF, N_ist, X_true

def load_HF_signals(path_data, N_ist, N_entries, N_obs):
    Recordings = np.load(path_data+'Recordings_MCMC.pkl',allow_pickle=True) #import osservazioni
    for i1 in range(N_ist*N_obs):
     Observations_HF = np.zeros((N_ist,N_obs,n_channels,N_entries)) #struttura per osservazioni 10x8x8x200
    for i1 in range(N_ist):
        for i2 in range(N_obs):
            for i3 in range(n_channels):
                Observations_HF[i1,i2,i3,:] = Recordings[i1*N_obs+i2,i3*N_entries:(i3+1)*N_entries] #reshape osservazioni
    return Observations_HF

X_HF, N_ist, X_true = read_HF_input(path_data_test) #carico input HF
Y_HF = load_HF_signals(work_path,N_ist//N_obs,N_entries,N_obs) #carico segnali HF obiettivo

#%%

#SETUP OF THE SURROGATE MODEL

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
#normalizzo osservazioni una volta per tutte: gaussiana standard per ogni canale
i1 = which_ist
for i8 in range(N_obs):
    for i9 in range(n_channels):
        Y_HF[i1,i8,i9,:] = (Y_HF[i1,i8,i9,:] - HF_signals_means[i9]) / HF_signals_stds[i9] 

#%%
#INITIALIZE SOME STRUCTURES

LF_signals_start = np.zeros((n_chains,N_entries,n_channels)) 
Input_HF_start = np.zeros((n_chains,N_entries,5+n_channels)) 
X_input_MF = np.zeros((n_chains,n_parameters+1)) #reshape da parametri campionati per il surrogato (+1 per tenere conto di damage_x e damage_y)

like_single_dof = np.zeros((n_chains,n_channels)) #contenitore per likelihood sul singolo sensore, per ogni osservazione
like_single_obs_now = np.zeros((n_chains,N_obs)) #contenitore per likelihood sulla singola osservazione
like_single_obs_before = np.zeros((n_chains,N_obs)) #come sopra, ma per iterazione precedente

Input_HF_start[:,:,0] = X_true[i1, 0] #forzare Frequency_true corrispondente a which_ist
Input_HF_start[:,:,1] = X_true[i1, 1] #frozare Amplitude_true corrispondente a  which_ist
Input_HF_start[:,:,4] = np.linspace(signal_resolution, signal_resolution * N_entries, N_entries) #timestep

X_input_MF[:,0] = X_true[i1, 0] #forzare Frequency_true corrispondente a which_ist
X_input_MF[:,1] = X_true[i1, 1] #frozare Amplitude_true corrispondente a  which_ist

#BEGIN MCMC
def RMSE(vect1, vect2):
    return np.sqrt(np.mean(np.square(vect1 - vect2)))

def RSSE(vect1, vect2):
    return np.sqrt(np.sum(np.square(vect1 - vect2)))

def single_param_like_single_obs(obs,mapping): #likelihood assumes independent prediction errors 
    for j1 in range(n_chains):
        for i1 in range(n_channels):
            rmse = RMSE(obs[i1], mapping[j1,i1])
            rsse = RSSE(obs[i1], mapping[j1,i1])
            like_single_dof[j1,i1] = 1./(np.sqrt(2.*np.pi)*rmse) * np.exp(-((rsse**2)/(2.*(rmse**2))))*1e35 #evitare underflow aritmetico
    return np.prod(like_single_dof,axis=1)


#valuto modello LF
LF_mapping_start = LF_net.predict(X_input_MF[:,0:2],verbose=0) #(n_chains,n_basi)

#normalizzo output LF indietro e riporto ai valori originali delle componenti principali
for i3 in range(N_basis):
    LF_mapping_start[:,i3]= LF_mean[i3]+(LF_mapping_start[:,i3]/weights[i3])*LF_std[i3]

#espando i segnali LF proiettando le comonenti principali sulla base
LF_reconstruct_start = np.matmul(basis,LF_mapping_start[:].T).T 
    
#normalizzo i segnali LF approssimati per darli in input alla rete HF
for i3 in range(n_channels): #normale standard su ogni canale
    LF_signals_start[:,:,i3] = (LF_reconstruct_start[:,i3*N_entries:(i3+1)*N_entries] - LF_signals_means[i3])/LF_signals_stds[i3]

for i3 in range(n_channels):
    Input_HF_start[:, :, 5 + i3] = LF_signals_start[:, :, i3]

#%%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
    
class MySurrogateModel_fine:
    def __init__(self, n_channels, signal_resolution, N_entries, LF_signals_start, HF_net_to_pred):
        self.n_channels = n_channels
        self.signal_resolution = signal_resolution #secondo me si può togliere
        self.N_entries = N_entries
        self.LF_signals_start = LF_signals_start
        self.HF_net_to_pred = HF_net_to_pred

    @tf.function(jit_compile=True,reduce_retracing=True)
    def predict_per_uno(self, Input_HF_start):
        # Controlla se l'input è un tensore, altrimenti lo converte
        if not isinstance(Input_HF_start, tf.Tensor):
            Input_HF_start = tf.convert_to_tensor(Input_HF_start, dtype=tf.float32)
            Input_HF_start.set_shape([self.n_chains,self.N_entries,5+self.n_channels])
        
        # Disabilita la tracciatura del gradiente per migliorare l'efficienza
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.stop_recording()  # Disabilita il calcolo dei gradienti
            # Previsione del segnale HF usando il modello pre-addestrato
            Y_start = self.HF_net_to_pred(Input_HF_start,training = False)

        return Y_start
    
    def __call__(self, parameters):
        # Definisci l'input per il modello surrogato
        if parameters <= 0.5:
            damage_x = parameters * 2.0
            damage_y = 0.0
        else:
            damage_x = 1.0
            damage_y = (parameters - 0.5) * 2.0

        
        Input_HF_start[:, :, 2] = damage_x  #riempio tensore di input per modello surrogato
        Input_HF_start[:, :, 3] = damage_y
        
        # Predici il segnale HF con la funzione ottimizzata
        Y_start = self.predict_per_uno(Input_HF_start)
        Y_flat = tf.reshape(Y_start, [-1])
        
        return Y_flat
    

class MySurrogateModel_coarse:
    def __init__(self, Input_HF_start):
        self.Input_HF_start = Input_HF_start
    
    def __call__(self, parameters):
        # Definisci l'input per il modello surrogato
        if parameters <= 0.5:
            damage_x = parameters * 2.0
            damage_y = 0.0
        else:
            damage_x = 1.0
            damage_y = (parameters - 0.5) * 2.0

        
        Input_HF_start[:, :, 2] = damage_x  #riempio tensore di input per modello surrogato
        Input_HF_start[:, :, 3] = damage_y
        Input_HF_start_flat = Input_HF_start.flatten()
    
        return Input_HF_start_flat

    
class custom_loglike_fine:
    def __init__(self, signal_resolution, n_channels, N_obs, N_entries, limit,Y_HF,LF_signals_start):
        self.signal_resolution = signal_resolution #secondo me si può togliere
        self.n_channels = n_channels
        self.N_obs = N_obs
        self.N_entries = N_entries
        self.limit = limit
        self.Y_HF = Y_HF
        self.LF_signals_start = LF_signals_start
        #self.elapsed = None
        #self.likelihoods_history = []  
    
    #def get_likelihoods_history(self):
        #return self.likelihoods_history 
    def get_elapsed(self):
       """Ritorna il tempo impiegato durante l'ultima predizione."""
       return self.elapsed

    def loglike(self,Y):
        
        #valuto likelihood: confronto output modello HF con osservazioni
        #start = time.time()
        Y_reshaped = Y.reshape(n_chains,N_entries,n_channels)
        for i2 in range(N_obs):
           like_single_obs_now[:,i2] = single_param_like_single_obs(self.Y_HF[i1,i2,:,limit:],np.transpose(Y_reshaped[:,limit:,:], axes=[0,2,1]))
        like_tot_now = np.prod(like_single_obs_now,axis=1)
        loglike_value = np.sum(np.log(like_tot_now))
        #self.likelihoods_history.append(np.exp(loglike_value))
        #self.elapsed = time.time() - start
        
        return loglike_value
    
class custom_loglike_coarse:
    def __init__(self,Y_HF):
        self.Y_HF = Y_HF
    
    def loglike(self,Input_HF):

        Input_HF = Input_HF.reshape(n_chains,N_entries,5+n_channels)
        Input_HF_r = Input_HF[0,0,0:4]
        Input_HF_r = Input_HF_r[np.newaxis,:]
        loglike_value_coarse = 0
        for i2 in range(N_obs):
            Y_HF_r = Y_HF[0,i2,:,:].T
            Y_HF_r = Y_HF_r[np.newaxis, :, :]
            predictions = Regressor.predict([Y_HF_r,Input_HF_r])
            predictions_true = predictions*6.3378+100.2235 
            loglike_value_coarse = loglike_value_coarse + predictions_true
    
        return loglike_value_coarse
 

class CustomUniform:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.area = self.upper_bound - self.lower_bound
    
    def pdf(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return 1 / self.area
        else:
            return 0
        
    def logpdf(self, x):   
        if self.pdf(x) == 0:
            return -np.inf
        else:
            return np.log(self.pdf(x))
    
    def rvs(self):
        return np.random.uniform(self.lower_bound, self.upper_bound)
    

my_prior = CustomUniform(0, 1)
my_loglike_coarse = custom_loglike_coarse(Y_HF)
my_loglike_fine = custom_loglike_fine(signal_resolution, n_channels, N_obs, N_entries, limit, Y_HF, LF_signals_start)
my_coarse_model = MySurrogateModel_coarse(Input_HF_start)
my_fine_model = MySurrogateModel_fine(n_channels, signal_resolution, N_entries, LF_signals_start, HF_net_to_pred)

# set up the link factories
my_posterior_coarse = tda.Posterior(my_prior, my_loglike_coarse, my_coarse_model)
my_posterior_fine = tda.Posterior(my_prior, my_loglike_fine, my_fine_model)
my_posteriors = [my_posterior_coarse, my_posterior_fine] 

# Set up the proposal : random walk Metropolis
rwmh_cov = 1e-2 *np.eye(1)
rwmh_adaptive = True
my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, adaptive=rwmh_adaptive)

# Run MCMC
import os
if "CI" in os.environ:
    iterations = 120
    burnin = 20
else:
    iterations = 500
    burnin = 50
    
my_chains = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=2, force_sequential=True)

#%%
import arviz as az
idata = tda.to_inference_data(my_chains, level='fine', burnin=burnin)  

params = idata.posterior["x0"].values

az.summary(idata)

az.plot_trace(idata)
plt.show()

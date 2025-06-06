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
import timeit

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
which_ist = 3 #tra 0 e 9


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

lik_min = -38.14900633995325#-653.6425949588155#
delta_max = 48.25939812179713#48.853169588398714 # 

def RMSE(vect1, vect2):
    return np.sqrt(np.mean(np.square(vect1 - vect2)))

def RSSE(vect1, vect2):
    return np.sqrt(np.sum(np.square(vect1 - vect2)))

# def single_param_like_single_obs(obs,mapping): #likelihood assumes independent prediction errors 
#     for j1 in range(n_chains):
#         for i1 in range(n_channels):
#             rmse = RMSE(obs[i1], mapping[j1,i1])
#             rsse = RSSE(obs[i1], mapping[j1,i1])
#             like_single_dof[j1,i1] = 1./(np.sqrt(2.*np.pi)*rmse)*np.exp (-((rsse**2)/(2.*(rmse**2)))) *1e35 #evitare underflow aritmetico
#     return np.prod(like_single_dof,axis=1)

def single_param_like_single_obs(obs,mapping): #likelihood assumes independent prediction errors 
    for j1 in range(n_chains):
        for i1 in range(n_channels):
            rmse = RMSE(obs[i1], mapping[j1,i1])
            rsse = RSSE(obs[i1], mapping[j1,i1])
            like_single_dof[j1,i1] = np.log(1./(np.sqrt(2.*np.pi)*rmse)) -((rsse**2)/(2.*(rmse**2)))
    return np.sum(like_single_dof,axis=1)


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
# MCMC 2 LIVELLI 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class ParameterHistory:
    def __init__(self, window_size=10000):
        self.history = []
        self.mean = 0.0
        self.std = 1.0
        self.window_size = window_size  # Mantiene solo gli ultimi 'window_size' valori

    def add(self, parameter):
        self.history.append(parameter)
        
        # # Mantieni solo gli ultimi 'window_size' valori
        # if len(self.history) > self.window_size:
        #     self.history.pop(0)

    def reset(self):
        self.history = []  # Azzerare la history quando necessario

    def get_mean_std(self):
        if len(self.history) == 0:
            return 0.0, 1.0  
        
        # Calcola media e std solo ogni tot iterazioni
        # if len(self.history) % 100 == 0:
        #     self.mean = np.mean(self.history[::100])
        #     self.std = np.std(self.history[::100])
        
        self.mean = np.mean(self.history)
        self.std = np.std(self.history)

        # Evita deviazione standard troppo piccola
        if self.std < 1e-6:
            self.std = 1e-6  
            
        return self.mean, self.std

    
param_history = ParameterHistory()
   
class MySurrogateModel_fine:
    def __init__(self, n_channels, N_entries, LF_signals_start, HF_net_to_pred,param_history):
        self.n_channels = n_channels
        self.N_entries = N_entries
        self.LF_signals_start = LF_signals_start
        self.HF_net_to_pred = HF_net_to_pred
        self.param_history = param_history
        self.iteration = 0  # Contatore di iterazioni

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
        
        self.iteration += 1
        if isinstance(parameters, np.ndarray) and parameters.ndim == 1:
            # Se 'parameter' è un array monodimensionale, estrai il valore scalare
            parameters = parameters.item()
        else: 
            parameters = parameters
        # print(f"Numero iterazioni fine: {self.iteration}")
        # print(f"parameter_fine: {parameters}")
            
    
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
    def __init__(self):
        self.iteration_coarse=0
    def __call__(self, parameters):
        
        self.iteration_coarse += 1
        #print(f"Numero iterazioni coarse: {self.iteration_coarse}")
        #print(f"parameter_coarse: {parameters}")
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
    def __init__(self,  n_channels, N_obs, N_entries, limit,Y_HF,LF_signals_start):
        self.n_channels = n_channels
        self.N_obs = N_obs
        self.N_entries = N_entries
        self.limit = limit
        self.Y_HF = Y_HF
        self.LF_signals_start = LF_signals_start

    def loglike(self,Y):
        
        Y_reshaped = Y.reshape(n_chains,N_entries,n_channels)
        for i2 in range(N_obs):
           like_single_obs_now[:,i2] = single_param_like_single_obs(self.Y_HF[i1,i2,:,limit:],np.transpose(Y_reshaped[:,limit:,:], axes=[0,2,1]))
        loglike_value = np.sum(like_single_obs_now,axis=1).item()
        
        return loglike_value

Y_HF_r = np.transpose(Y_HF, (0, 1, 3, 2)) #(10,8,200,8)


class custom_loglike_coarse:
    def __init__(self,Y_HF_r,Regressor,param_history):
        self.Y_HF_r = Y_HF_r
        self.Regressor = Regressor
        self.param_history = param_history
    
    @tf.function(jit_compile=True, reduce_retracing=True)
    def predict_optimized(self, Y_HF_r, Input_HF_r):
            
            predictions = self.Regressor([self.Y_HF_r[i1], Input_HF_r[:, 0, 0:4]], training=False)
            
            return predictions
        
    
    def loglike(self,Input_HF):
        
        Input_HF = tf.reshape(Input_HF, [n_chains,N_entries, 5 + n_channels])

        Input_HF_r = tf.tile(Input_HF, [N_obs, 1, 1])  # Replica Input_HF
        start_time = timeit.default_timer()
        predictions = self.predict_optimized(Y_HF_r,Input_HF_r)
        elapsed_time = timeit.default_timer() - start_time
        predictions_true = predictions*delta_max+lik_min
        loglike_value_coarse = tf.reduce_sum(predictions_true) 

        
      
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
my_loglike_coarse = custom_loglike_coarse(Y_HF_r,Regressor,param_history)
my_loglike_fine = custom_loglike_fine(n_channels, N_obs, N_entries, limit, Y_HF, LF_signals_start)
my_coarse_model = MySurrogateModel_coarse()
my_fine_model = MySurrogateModel_fine(n_channels, N_entries, LF_signals_start, HF_net_to_pred,param_history)

# set up the link factories
my_posterior_coarse = tda.Posterior(my_prior, my_loglike_coarse, my_coarse_model)
my_posterior_fine = tda.Posterior(my_prior, my_loglike_fine, my_fine_model)
my_posteriors = [my_posterior_coarse, my_posterior_fine] 

# Set up the proposal : random walk Metropolis
rwmh_cov = 1e-2*np.eye(1)
rwmh_adaptive = True
#my_proposal = tda.GaussianRandomWalk(C=rwmh_cov, adaptive=rwmh_adaptive)

# Adaptive Metropolis -> modificare file utils di tinyda per farlo funzionare
am_cov = 1e-2*np.eye(1)
am_t0 = 500
am_sd = 1
am_epsilon = 1e-6
am_adaptive = True
my_proposal = tda.AdaptiveMetropolis(C0=am_cov, t0=am_t0, sd=am_sd, epsilon=am_epsilon, adaptive=am_adaptive)

# Run MCMC
import os
if "CI" in os.environ:
    iterations = 120
    burnin = 1
else:
    iterations = 4000
    burnin = 400
    
my_chains = tda.sample(my_posteriors, my_proposal, iterations=iterations, n_chains=5,subsampling_rate=1,store_coarse_chain=False,force_sequential=True)
#my_chains = tda.sample(my_posterior_fine, my_proposal, iterations=iterations, n_chains=2,force_sequential=True)

#%%
import arviz as az
#idata = tda.to_inference_data(my_chains,level='fine',burnin=400) 
idata = tda.to_inference_data(my_chains,burnin=burnin) 
 


new_idata = idata.sel(draw=slice(None, None, 4), groups="posterior")

params = new_idata.posterior["x0"].values

print(az.summary(new_idata))
az.plot_trace(new_idata)


plt.show()

# #Effective Sample Size
# ess = az.ess(new_idata,var_names=["x0"])
# print(ess)


#%%

#unique_chain = params.reshape(-1, order='F')
params = np.load('C:\\Users\\Ruggi\\Carlotta\\FOM.npy')

unique_chain = params.reshape(-1)

mean_val = np.mean(unique_chain)


# Crea il trace plot
plt.figure(figsize=(10, 5))

plt.plot(
    unique_chain[0:1400]*7.4+0.15,
    alpha = 0.6,
    linewidth = 0.8,
    color = 'coral', label='Markov chain')

plt.legend()
target_val = X_HF[i1*N_obs,0]/2
# plt.axhline(target_val*7.4+0.15, color='navy', linewidth=2, label=r'Target value of $\theta_\Omega$')
# plt.axhline(mean_val*7.4+0.15, color='darkmagenta', linewidth=2, label='Posterior mean')

# Linea orizzontale "target" limitata all'intervallo x
plt.plot([0, 1399], [target_val*7.4+0.15,target_val*7.4+0.15],
         color='navy', linewidth=2, label=r'Target value of $\theta_\Omega$')

# Linea orizzontale "posterior mean" limitata all'intervallo x
plt.plot([0, 1399], [mean_val*7.4+0.15, mean_val*7.4+0.15],
         color='darkmagenta', linewidth=2, label='Posterior mean')

# Calcolo dell'intervallo di confidenza al 95%
lower_95 = np.percentile(unique_chain[0:1400]*7.4+0.15, 2.5)
upper_95 = np.percentile(unique_chain[0:1400]*7.4+0.15, 97.5)

plt.fill_between(
    x=np.arange(len(unique_chain[0:1400]*7.4+0.15)),
    y1=lower_95,
    y2=upper_95,
    color='powderblue',
    alpha=0.4,
    label='95% CI'
)

lower_75 = np.percentile(unique_chain[0:1400]*7.4+0.15, 12.5)
upper_75 = np.percentile(unique_chain[0:1400]*7.4+0.15, 87.5)

plt.fill_between(
    x=np.arange(len(unique_chain[0:1400]*7.4+0.15)),
    y1=lower_75,
    y2=upper_75,
    color='lightskyblue',
    alpha=0.6,
    label='75% CI'
)


lower_50 = np.percentile(unique_chain[0:1400]*7.4+0.15, 25)
upper_50 = np.percentile(unique_chain[0:1400]*7.4+0.15, 75)

plt.fill_between(
    x=np.arange(len(unique_chain[0:1400]*7.4+0.15)),
    y1=lower_50,
    y2=upper_50,
    color='steelblue',
    alpha=0.6,
    label='50% CI'
)

plt.ylim(0.5, 7.4)

#plt.title("Trace Plot - Multiple Chains")
plt.xlabel("Discrete step",fontsize=16)
plt.ylabel(r"Sample of $p(\theta_\Omega \mid \mathbf{y}^{\mathrm{exp}}_{1,\ldots,N_{\mathrm{obs}}})$",fontsize=16)
plt.legend(
    loc='upper center',     # posizione centrata in alto
    #bbox_to_anchor=(0.5, 1.15),  # sposta sopra il grafico
    ncol=2,                 # distribuisce le voci su due colonne (opzionale)
    fontsize=17,            # dimensione del font
    frameon=False           # rimuove il bordo della legenda (opzionale)
)
plt.show()

#%%
n_chains = 5
# Configurazione del plot
plt.figure(figsize=(10, 5))
plt.title("Distribuzione del Posterior (KDE)", fontsize=14)
plt.xlabel("Valore del parametro", fontsize=12)
plt.ylabel("Densità", fontsize=12)
plt.grid(False)

# Colori per le catene
#colors = plt.cm.viridis(np.linspace(0, 1, n_chains))
from scipy.stats import gaussian_kde
# 1. KDE per ogni catena (linee separate)
for chain_idx in range(n_chains):
    samples = params[chain_idx, :]
    kde = gaussian_kde(samples)
    x_vals = np.linspace(samples.min(), samples.max(), 500)
    y_vals = kde(x_vals)
    
    plt.plot(x_vals, y_vals, 
             #color=colors[chain_idx], 
             alpha=0.8,
             linewidth=2,
             label=f"Catena {chain_idx+1}")
plt.legend()

#%%
combined_samples = params.reshape(-1)  # Assumendo params shape (n_chains, n_samples_per_chain)
kde = gaussian_kde(combined_samples)
x_vals = np.linspace(combined_samples.min(), combined_samples.max(), 500)
y_vals = kde(x_vals)

plt.figure(figsize=(10, 5))
plt.title("Distribuzione del Posterior combinata (KDE)", fontsize=14)
plt.xlabel("Valore del parametro", fontsize=12)
plt.ylabel("Densità", fontsize=12)
plt.grid(False)
plt.plot(x_vals, y_vals, color='black', linewidth=2.5, label="Posterior combinato")
plt.legend()



#%%

if np.mean(params) <= 0.5:
    coord_x_finded = 0.15 + np.mean(params)*2*3.7
    coord_y_finded = 0.15
else:
    coord_x_finded = 3.85
    coord_y_finded = 0.15 + (np.mean(params)-0.5)*2*3.7  # corretto np.means -> np.mean


#unique_chain = params.reshape(-1, order='F')


if X_HF[i1*N_obs,1]==0:   
    target = X_HF[i1*N_obs,0]/2.
    coord_x_target = 0.15 + X_HF[i1*N_obs,0] * 3.7
    coord_y_target = 0.15
else:
    target = 0.5 + X_HF[i1*N_obs,1]/2.
    coord_x_target = 3.85
    coord_y_target = 0.15 + X_HF[i1*N_obs,1] * 3.7


step = 0.025
hist, bin_edges = np.histogram(unique_chain, bins=np.arange(0, 1, step), density=True)
coord_x = np.zeros((len(hist)))
coord_y = np.zeros((len(hist)))
coord_z = np.zeros((len(hist))) + 0.0775
dx = np.zeros((len(hist)))
dy = np.zeros((len(hist)))
dz = hist/hist.max()*3

for i10 in range(len(hist)):
    if i10 < len(hist)//2:
        coord_y[i10] = 0.05
        coord_x[i10] = bin_edges[i10]*7.85+0.05
        dx[i10] = step * 8 - 0.1
        dy[i10] = 0.2
    else:
        coord_y[i10] = bin_edges[i10]*7.85-3.7 + 0.05
        coord_x[i10] = 3.75
        dy[i10] = step * 8 - 0.1
        dx[i10] = 0.2


# vertices of a pyramid
v = np.array([[0,0,0], [4,0,0], [4,4,0], [3.7,4,0], [3.7,0.3,0], [0,0.3,0], [0,0,-0.4], [4,0,-0.4], [4,4,-0.4], [3.7,4,-0.4], [3.7,0.3,-0.4], [0,0.3,-0.4], 
              [0,-0.4,0.4], [0,0.8,0.4], [0,-0.4,-0.8], [0,0.8,-0.8], [coord_x_target-0.15, coord_y_target-0.15, 0], [coord_x_target-0.15, coord_y_target+0.15, 0], 
              [coord_x_target+0.15, coord_y_target-0.15, 0], [coord_x_target+0.15, coord_y_target+0.15, 0], [coord_x_target-0.15, coord_y_target-0.15, -0.4], 
              [coord_x_target-0.15, coord_y_target+0.15, -0.4], [coord_x_target+0.15, coord_y_target-0.15, -0.4], [coord_x_target+0.15, coord_y_target+0.15, -0.4], 
              [coord_x_finded-0.15, coord_y_finded-0.15, 0], [coord_x_finded-0.15, coord_y_finded+0.15, 0], [coord_x_finded+0.15, coord_y_finded-0.15, 0], 
              [coord_x_finded+0.15, coord_y_finded+0.15, 0], [coord_x_finded-0.15, coord_y_finded-0.15, -0.4], [coord_x_finded-0.15, coord_y_finded+0.15, -0.4], 
              [coord_x_finded+0.15, coord_y_finded-0.15, -0.4], [coord_x_finded+0.15, coord_y_finded+0.15, -0.4]])

# generate list of sides' polygons of our pyramid
verts = [[v[0],v[1],v[7],v[6]], [v[1],v[2],v[8],v[7]], [v[2],v[3],v[9],v[8]], [v[3],v[4],v[10],v[9]], [v[4],v[5],v[11],v[10]], [v[0],v[5],v[11],v[6]], [v[0],v[1],v[2],v[3],v[4],v[5]], [v[6],v[7],v[8],v[9],v[10],v[11]]]
verts_2 = [[v[12], v[13], v[15], v[14]]]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# ---- Sposta il pavimento PRIMA delle barre 3D ----
y_floor = np.linspace(0, 4, 10)
x_floor = np.linspace(0, 4, 10)
Xf, Yf = np.meshgrid(x_floor, y_floor)
Zf = np.zeros_like(Xf)
ax.plot_surface(Xf, Yf, Zf, color='lightgray', alpha=0.1)



# Prima disegna base/parallelepipedi
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2], s=0)
ax.add_collection3d(Poly3DCollection(verts, facecolors='steelblue', linewidths=1, edgecolors='dimgrey', alpha=.2))
ax.add_collection3d(Poly3DCollection(verts_2, facecolors='black', linewidths=1, edgecolors='black', alpha=.7))

# Poi disegna barre d’oro (istogramma)
ax.bar3d(coord_x, coord_y, coord_z, dx, dy, dz,
        color='gold', edgecolor='black', linewidth=0.3, alpha=1, shade=True)



# View angle
ax.view_init(elev=35, azim=310)

# Completely hide the Z-axis
ax.set_zticks([])
ax.set_zticklabels([])
ax.set_zlabel('')

# Make all background panes (walls) invisible
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # XZ plane (left wall)
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # YZ plane (right wall)
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # XY plane (keep floor visible via plot_surface)

# Hide the axis lines if desired
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Adjust remaining visible axes
ax.set_xlabel('[m]')
ax.set_ylabel('[m]')
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, np.max(dz)*1.2)
ax.grid(False)
ax.set_box_aspect([1,1,0.6])

# Disable the 3D background grid
ax.set_axis_off()  # This hides all axes and panes completely
# Then manually re-enable just what you want to show
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)
ax.xaxis.set_tick_params(labelbottom=True)
ax.yaxis.set_tick_params(labelleft=True)

# Offsets per centrare gli assi sugli spigoli della maniglia
offset_x = 0.22   # Sposta l’asse X a destra
offset_y = 4.06 # Sposta l’asse Y a destra (verso x=4.05)

tick_length = 0.04

#Asse X esterno: da sinistra a destra, spostato verso +x
ax.plot([0 + offset_x, 4 + offset_x], [-0.3, -0.3], [0, 0], color='black', linewidth=1.5)
for xtick in np.arange(0, 4.5, 0.5):
    ax.plot([xtick + offset_x, xtick + offset_x], [-0.3 - tick_length, -0.3 + tick_length], [0, 0],
            color='black', linewidth=1)
    ax.text(xtick + offset_x, -0.35 - 4 * tick_length, 0, f"{xtick:.1f}",
            ha='center', va='top', fontsize=8)


# Asse Y esterno: da basso verso alto, spostato verso +x (lato opposto)
# Lo abbassiamo con un offset z = -0.4
z_offset_y = -0.4
ax.plot([offset_y, offset_y], [0, 4], [z_offset_y, z_offset_y], color='black', linewidth=1.5)
for ytick in np.arange(0, 4.5, 0.5):
    ax.plot([offset_y - tick_length, offset_y + tick_length], [ytick, ytick], [z_offset_y, z_offset_y],
            color='black', linewidth=1)
    ax.text(offset_y + 4 * tick_length, ytick, z_offset_y, f"{ytick:.1f}",
            ha='left', va='center', fontsize=8)

x_center = offset_x + 2  # metà di 0-4 + offset_x
y_pos_x_label = -0.35 - 13 * tick_length  # più distante rispetto alle etichette numeriche

ax.text(x_center, y_pos_x_label, 0, '[m]', ha='center', va='top', fontsize=9, fontweight='bold')

# Range asse Y: da 0 a 4 (con offset_y)
y_center = 2  # metà di 0-4
x_pos_y_label = offset_y + 11 * tick_length  # più distante rispetto alle etichette numeriche
z_pos_y_label = z_offset_y

ax.text(x_pos_y_label, y_center, z_pos_y_label, '[m]', ha='left', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()




#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Caricamento dei dati
params = np.load('C:\\Users\\Ruggi\\Carlotta\\FOM.npy')
unique_chain = params.reshape(-1)
mean_val = np.mean(unique_chain)
target_val = X_HF[i1 * N_obs, 0] / 2  # Definisci X_HF, i1, N_obs prima

# -------------------------------
# Setup della figura con due pannelli
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -------------------------------
# Sinistra: Trace Plot
# -------------------------------
ax = axes[0]
trace_vals = unique_chain[0:1400]*7.4 + 0.15

ax.plot(trace_vals, alpha=0.6, linewidth=0.8, color='coral', label='Markov chain')

# Linee orizzontali target e media posteriore
ax.plot([0, 1399], [target_val*7.4+0.15]*2, color='navy', linewidth=2, label=r'Target value of $\theta_\Omega$')
ax.plot([0, 1399], [mean_val*7.4+0.15]*2, color='darkmagenta', linewidth=2, label='Posterior mean')

# Intervalli di confidenza
for lower, upper, color, alpha, label in [
    (2.5, 97.5, 'powderblue', 0.4, '95% CI'),
    (12.5, 87.5, 'lightskyblue', 0.6, '75% CI'),
    (25, 75, 'steelblue', 0.6, '50% CI')
]:
    l = np.percentile(trace_vals, lower)
    u = np.percentile(trace_vals, upper)
    ax.fill_between(np.arange(len(trace_vals)), y1=l, y2=u, color=color, alpha=alpha, label=label)

ax.set_ylim(0.5, 7.4)
ax.set_xlabel("Discrete step", fontsize=12)
ax.set_ylabel(r"$p(\theta_\Omega \mid \mathbf{y}^{\mathrm{exp}}_{1,\ldots,N_{\mathrm{obs}}})$", fontsize=14)
ax.legend(loc='upper center', ncol=2, fontsize=11, frameon=False)
ax.set_title("Trace Plot", fontsize=14)

# -------------------------------
# Destra: KDE del Posterior
# -------------------------------
ax = axes[1]
combined_samples = unique_chain
kde = gaussian_kde(combined_samples)
x_vals = np.linspace(combined_samples.min(), combined_samples.max(), 500)
y_vals = kde(x_vals)

ax.plot(x_vals, y_vals, color='black', linewidth=2.5, label="Posterior combinato")
ax.set_xlabel("Valore del parametro", fontsize=12)
ax.set_ylabel("Densità", fontsize=12)
ax.set_title("Distribuzione del Posterior (KDE)", fontsize=14)
ax.grid(False)
ax.legend()

plt.tight_layout()
plt.show()





#%%
# Y_HF_r = np.transpose(Y_HF, (0, 1, 3, 2)) #(10,8,200,8)

# #OTTIMIZZAZIONE

# from scipy.optimize import minimize

# def neg_log_likelihood_fine(theta):
#     theta = theta[0]
#     # Calcola il danno strutturale in base a theta
#     if theta <= 0.5:
#         damage_x = theta * 2.0
#         damage_y = 0.0
#     else: 
#         damage_x = 1.0
#         damage_y = (theta - 0.5) * 2.0

#     # Aggiorna il tensore di input per il modello surrogato
#     Input_HF_start[:, :, 2] = damage_x  
#     Input_HF_start[:, :, 3] = damage_y  

#     # Predici il segnale HF con il modello surrogato
#     Y_start = HF_net_to_pred.predict(Input_HF_start, verbose=0)

#     # Calcola la log-likelihood
#     like_single_obs_now = np.zeros((Y_start.shape[0], N_obs))
#     for i2 in range(N_obs):
#         like_single_obs_now[:, i2] = single_param_like_single_obs(
#             Y_HF[i1, i2, :, limit:], 
#             np.transpose(Y_start[:, limit:, :], axes=[0, 2, 1])
#         )
    
#     like_tot_now = np.prod(like_single_obs_now, axis=1)
#     loglike_value = np.sum(np.log(like_tot_now))

#     return -loglike_value  # Minimizzazione della log-likelihood negativa

# # Valore iniziale per theta
# theta_init = np.array([0.5])  

# # Esegui l'ottimizzazione con scipy.optimize.minimize
# result_fine = minimize(neg_log_likelihood_fine, theta_init, method='Nelder-Mead')

# # Stampa i risultati
# print("Theta ottimizzato livello fine:", result_fine.x)


# def neg_log_likelihood_coarse(theta):
    
#     theta = theta[0]
#     # Calcola il danno strutturale in base a theta
#     if theta <= 0.5:
#         damage_x = theta * 2.0
#         damage_y = 0.0
#     else: 
#         damage_x = 1.0
#         damage_y = (theta - 0.5) * 2.0
    
#     Input_HF_start[:, :, 2] = damage_x  #riempio tensore di input per modello surrogato
#     Input_HF_start[:, :, 3] = damage_y
    
#     Input_HF_r = np.tile(Input_HF_start, (N_obs, 1, 1))
#     predictions = Regressor.predict([Y_HF_r[i1],Input_HF_r[:,0,0:4]],verbose=0)
#     predictions_true = predictions*delta_max+lik_min
#     loglike_value_coarse = np.sum(predictions_true)
    
#     return -loglike_value_coarse

# # Valore iniziale per theta
# theta_init = np.array([0.5])  

# # Esegui l'ottimizzazione con scipy.optimize.minimize
# result_coarse = minimize(neg_log_likelihood_coarse, theta_init, method='Nelder-Mead')

# # Stampa i risultati
# print("Theta ottimizzato livello coarse:", result_coarse.x)

#%%

cmap = plt.get_cmap('tab20c')  # Ottieni una colormap
colors = [cmap(i) for i in range(n_chains)]  # Assegna un colore per catena
n_chains = 1
samples = params

for chain_idx in range(n_chains):
    plt.plot(
        samples[chain_idx, :],  # Samples della catena corrente
        alpha=0.9,                # Trasparenza per sovrapposizione
        linewidth = 1.5,
        color=colors[chain_idx],
        label=f"Chain {chain_idx + 1}"
    )

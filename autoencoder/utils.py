# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:20:17 2024

@author: Ruggi
"""
import numpy as np

def load_HF_signals(file_paths,N_ist,n_channels,N_entries):
    """
    Elabora i dati dai file CSV e restituisce una matrice strutturata delle osservazioni.
    
    Args:
        file_paths (list of str): Lista di percorsi ai file CSV da leggere.
        n_instances (int): Numero di istanze.
        n_entries (int): Numero di osservazioni per canale.
        n_channels (int): Numero di canali.

    Returns:
        np.ndarray: Matrice delle osservazioni con forma (n_instances, n_channels, n_entries).
    """
    # Leggi i dati dai file CSV
    data = [np.genfromtxt(path) for path in file_paths]

    # Combina i dati in una matrice
    U_concat = np.column_stack(data)
    
    # Rimuovo la prima riga ad ogni istanza, poichè formata da valori nulli
    U_concat_ridotta = np.empty((N_ist * (N_entries), n_channels))

    for i in range(N_ist):
        U_concat_ridotta[i * (N_entries):(i + 1) * (N_entries), :] = U_concat[i * 201 + 1:(i + 1) * 201, :]
    
    #struttura per osservazioni 1000x8x200 (istanze,canali,entries)
    Y_HF = np.zeros((N_ist,n_channels,N_entries))
    Y_HF = U_concat_ridotta.reshape(N_ist, N_entries, n_channels).transpose(0, 2, 1)
    
    return Y_HF


def add_noise_to_dataset(Y_HF, SNR):
    """
    Aggiunge rumore gaussiano a un dataset in base al rapporto segnale-rumore (SNR).
    
    Args:
        Y_HF (np.ndarray): Dataset originale con forma (N_ist, n_channels, N_entries).
        SNR (float): Rapporto segnale-rumore (Signal-to-Noise Ratio).
        
    Returns:
        np.ndarray: Dataset con rumore aggiunto, stessa forma di Y_HF.
    """
    N_ist, n_channels, N_entries = Y_HF.shape
    Y_exp = np.zeros((N_ist,n_channels,N_entries))

    for i1 in range(N_ist):
        for i2 in range(n_channels):
            signal_power = np.var(Y_HF[i1, i2, :])  # Potenza del segnale
            noise_power = signal_power / SNR       # Potenza del rumore
            noise = np.random.normal(0, np.sqrt(noise_power), Y_HF[i1, i2, :].shape)
            Y_exp[i1, i2, :] = Y_HF[i1, i2, :] + noise  # Aggiungi rumore

    return Y_exp


def normalize_dataset(Y_exp, media=None, std=None):
    """
    Normalizza un dataset per canale.
    Se media e deviazione standard sono forniti, li utilizza per normalizzare.
    Altrimenti, calcola media e deviazione standard dal dataset.
    
    Args:
        Y_exp (np.ndarray): Dataset da normalizzare, con forma (N_ist, n_channels, N_entries).
        media (np.ndarray, opzionale): Media per canale. Se `None`, viene calcolata da `Y_exp`.
        std (np.ndarray, opzionale): Deviazione standard per canale. Se `None`, viene calcolata da `Y_exp`.
    
    Returns:
        tuple: (Y_exp_norm, media, std)
            - Y_exp_norm: Dataset normalizzato.
            - media: Media per canale usata per la normalizzazione.
            - std: Deviazione standard per canale usata per la normalizzazione.
    """
    # Recupera le dimensioni
    N_ist, n_channels, N_entries = Y_exp.shape
    
    # Ristruttura i dati per concatenare le istanze in una matrice per canale
    Y_exp_new = Y_exp.reshape(N_ist,n_channels,N_entries).transpose(1, 0, 2).reshape(n_channels, -1)
    
    # Calcola media e deviazione standard solo se non fornite
    if media is None or std is None:
        media = np.mean(Y_exp_new, axis=1)
        std = np.std(Y_exp_new, axis=1)
    
    # Normalizza il dataset
    for i1 in range(n_channels):
        Y_exp_new[i1,:] = (Y_exp_new[i1,:]-media[i1])/std[i1]
        
    Y_exp_norm = Y_exp_new.reshape(n_channels, N_ist, N_entries).transpose(1, 0, 2)
    
    return Y_exp_norm, media, std



def apply_surrogate_model(X_HF, indices, n_ist_par, N_entries, n_channels, 
                          signal_resolution, LF_net, HF_net_to_pred, 
                          LF_mean, LF_std, weights, basis, LF_signals_means, LF_signals_stds):
    """
    Applica il modello surrogato per generare Y_i su un dataset.
    
    Parametri:
    - X_HF: array di input dei parametri HF.
    - indices: indici di train o test (train_indices/test_indices).
    - n_ist_par: numero di istanze di parametri.
    - N_entries: numero di entry per istanza.
    - n_channels: numero di canali.
    - signal_resolution: passo temporale del segnale.
    - LF_net: rete neurale per il modello a bassa fedeltà.
    - HF_net_to_pred: rete neurale per il modello ad alta fedeltà.
    - LF_mean, LF_std: parametri di normalizzazione per il modello LF.
    - weights: pesi delle componenti principali.
    - basis: matrice di base per la ricostruzione dei segnali LF.
    - LF_signals_means, LF_signals_stds: medie e deviazioni standard dei segnali LF.
    
    Restituisce:
    - Y: array delle uscite generate dal modello surrogato.
    - Input_HF: parametri del modello high fidelity
    """
    N_ist = len(indices)  # Numero di istanze (train o test)
    
    # Strutture dati per input e output
    X_HF_selected = np.zeros((n_ist_par, N_ist, 4))
    for i1 in range(n_ist_par):
        X_HF_selected[i1] = X_HF[i1, indices, :]
    
    Input_HF = np.zeros((n_ist_par, N_ist, N_entries, 5 + n_channels))
    X_input_MF = np.zeros((n_ist_par, N_ist, 2))  # Per LF_net: 2 parametri
    Y = np.zeros((n_ist_par, N_ist, n_channels, N_entries))  # Output finale
    
    # Ciclo principale
    for i1 in range(n_ist_par):
        # Valutazione modello LF
        X_input_MF[i1, :, 0] = X_HF_selected[i1, :, 0]
        X_input_MF[i1, :, 1] = X_HF_selected[i1, :, 1]
        LF_mapping_start = LF_net.predict(X_input_MF[i1, :, 0:2], verbose=0)
        
        # Normalizzazione inversa e ricostruzione segnali LF
        for i3 in range(LF_mapping_start.shape[1]):
            LF_mapping_start[:, i3] = LF_mean[i3] + (LF_mapping_start[:, i3] / weights[i3]) * LF_std[i3]
        
        #Espando i segnali LF proiettando le componenti principali sulla base
        LF_reconstruct_start = np.matmul(basis, LF_mapping_start[:].T).T
        
        # Normalizzazione segnali LF per l'input HF
        LF_signals_start = np.zeros((N_ist, N_entries, n_channels))
        for i3 in range(n_channels):
            LF_signals_start[:, :, i3] = (LF_reconstruct_start[:, i3 * N_entries:(i3 + 1) * N_entries] - LF_signals_means[i3]) / LF_signals_stds[i3]
            Input_HF[i1, :, :, 5 + i3] = LF_signals_start[:, :, i3]
        
        # Valutazione modello HF
        for i3 in range(N_ist):
            Input_HF[i1, i3, :, 0] = X_HF_selected[i1, i3, 0]  # Frequenza
            Input_HF[i1, i3, :, 1] = X_HF_selected[i1, i3, 1]  # Ampiezza
            Input_HF[i1, i3, :, 2] = X_HF_selected[i1, i3, 2]  # Coord_x danno
            Input_HF[i1, i3, :, 3] = X_HF_selected[i1, i3, 3]  # Coord_y danno
            Input_HF[i1, i3, :, 4] = np.linspace(signal_resolution, signal_resolution * N_entries, N_entries)  # Timestep
        
        Y_i = HF_net_to_pred.predict(Input_HF[i1, :, :, :], verbose=0)
        Y[i1] = np.transpose(Y_i, (0, 2, 1))
    
    return Y,Input_HF


def RMSE(vect1, vect2):
    """Calcola la Root Mean Square Error tra due vettori."""
    return np.sqrt(np.mean(np.square(vect1 - vect2)))

def RSSE(vect1, vect2):
    """Calcola la Root Sum Square Error tra due vettori."""
    return np.sqrt(np.sum(np.square(vect1 - vect2)))

def compute_likelihood(Y_exp, Y, N_ist, n_ist_par, n_channels, N_entries, removed_ratio, limit):
    """
    Calcola la log-likelihood tra due set di dati: uno esperimentale (Y_exp) e uno predetto (Y).
    
    Parametri:
    - Y_exp: dati esperimentali (Y_exp_train o Y_exp_test)
    - Y: dati predetti dal modello (Y_train o Y_test)
    - N_ist: numero di istanze nel set (training o test)
    - n_ist_par: numero di istanze di parametri
    - n_channels: numero di canali
    - N_entries: numero di time steps per istanza
    - removed_ratio: frazione di time steps da rimuovere nel calcolo della likelihood (default 0.2)
    
    Restituisce:
    - likelihood: matrice di log-likelihood
    """
    likelihood = np.zeros((N_ist, n_ist_par))  # Inizializzazione della matrice di likelihood
    
    # Ciclo su tutte le istanze e calcolo la log-likelihood
    for i1 in range(N_ist):
        for i2 in range(n_ist_par):
            somma = 0
            for i3 in range(n_channels):
                rmse = RMSE(Y_exp[i1, i3, limit:], Y[i2, i1, i3, limit:])#*10
                rsse = RSSE(Y_exp[i1, i3, limit:], Y[i2, i1, i3, limit:])
                somma += np.log(1. / (np.sqrt(2. * np.pi) * rmse)) + (-((rsse ** 2) / (2. * (rmse ** 2)))) 
            likelihood[i1, i2] = somma  # Log-likelihood per l'istanza i1 e parametro i2
    
    return likelihood


def likelihood_reshaped(likelihood, N_ist, n_ist_par):
    """
    Funzione per reshaping della matrice di likelihood.
    
    Parametri:
    - likelihood: matrice di likelihood (ad esempio likelihood_train o likelihood_test)
    - N_ist: numero di istanze nel set (training o test)
    - n_ist_par: numero di istanze di parametri
    
    Restituisce:
    - likelihood_r: matrice reshaped
    """
    likelihood_r = np.reshape(likelihood, (N_ist * n_ist_par,),'F')
    return likelihood_r
    
    
    
def reshape_input_HF(Input_HF, N_ist, n_ist_par):
    """
    Funzione per reshaping della matrice Input_HF_train.
    
    Parametri:
    - Input_HF_train: matrice di Input_HF_train
    - N_ist_train: numero di istanze nel set di addestramento
    - n_ist_par: numero di istanze di parametri
    
    Restituisce:
    - Input_HF_r_train: matrice reshaped di Input_HF_train
    """
    Input_HF_r = np.zeros((N_ist * n_ist_par, 4))
    for i in range(n_ist_par):
        Input_HF_r[i*N_ist:(i+1)*N_ist,:] = Input_HF[i,:,0,0:4]
    return Input_HF_r


def reshape_Y_exp(Y_exp, N_ist, n_ist_par, n_channels, N_entries):
    """
    Funzione per reshaping della matrice Y_exp_train.
    
    Parametri:
    - Y_exp_train: matrice di Y_exp_train
    - N_ist_train: numero di istanze nel set di addestramento
    - n_ist_par: numero di istanze di parametri
    - n_channels: numero di canali
    - N_entries: numero di entries per istanza
    
    Restituisce:
    - Y_exp_r_train: matrice reshaped di Y_exp_train
    """
    Y_exp_r = np.zeros((N_ist * n_ist_par, n_channels, N_entries))
    for i in range(n_ist_par):
        start = i * N_ist
        end = (i + 1) * N_ist
        Y_exp_r[start:end, :, :] = Y_exp
    
    Y_exp_r = np.transpose(Y_exp_r, (0, 2, 1))
    return Y_exp_r
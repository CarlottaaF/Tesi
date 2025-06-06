# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:51:11 2025

@author: Ruggi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#%%

ess_one_level = [2353,1997,2396,2508,2251,2635,2082,2358,2365,2483,2597,2502,2367,
                 2398,2679,2432,2713,2184,2009,2230,2397,2439,2465,2510,2623]  
ess_two_level = [799.2,524.9,561,609.7,598.2,495,952.7,334.6,607.3,869.9,878.5,518.9,
                 730.1,320.3,1192,1006,708.4,231.9,686.2,946.6,1028,525.8,219.2,
                 896.6,639.4]
#ess_2levels_AM = {924.7,816.2,788.5,961.0,946.0,288.7,766.2,1114,877.4,895.8,575.3,643,876.9,597.7,951.1,592.2,
                  #712.8,748.8,785.1,1237,890.9,586.4,1155,933.8,940.3}
ess_2levels_AM = {696.6,770.4,797.8,570.3,824.3,439,704.9,387.4,878.9,939.9,681.1,802.7,462.0,178.5,963.2,1273,924.4,507.2,810.1,
                  1188,721.1,748.2,653.8,469.3,1049}


time_one_level = [14.57,12.58,12.7,12.69,12.69,12.66,12.56,16.42,16.26,13.63,12.73,
                  12.52,12.52,12.51,12.54,12.51,12.53,12.53,12.54,12.56,12.52,12.53,
                  12.52,12.54,12.53]  
time_two_levels = [6.13,6.06,6.79,6.75,6.87,6.84,5.13,5.61,5.12,6,6.02,5.16,6.05,5.61,
                  5.58,6.20,6.05,6.30,6.28,6.19,6.34,6.13,5.68,5.05,5.59]
#time_2levels_AM = [5.10,5.9,5.15,5.10,5.13,5.6,5.1,6.06,5.58,6.12,5.13,6.06,6.07,6.17,
                   #6.18,6.14,5.07,5.04,5.06,5,5.11,5.58,5.13,5.1,6.02]
time_2levels_AM = [5.16,5.15,5.09,5.07,5.09,5.59,5.16,5.10,6,5.1,5.58,5.15,6.05,6.05,5.68,5.13,5.55,5.12,5.13,5.15,5.15,
                      5.6,6.05,6.01,5.55]


posterior_one_level = [0.391,0.393,0.389,0.386,0.392,0.391,0.388,0.393,0.389,0.39,
                       0.389,0.391,0.391,0.394,0.39,0.39,0.392,0.393,0.392,0.393,
                       0.392,0.392,0.39,0.391,0.385]
posterior_two_level = [0.39,0.388,0.381,0.389,0.386,0.382,0.388,0.387,0.387,0.393,
                       0.391,0.388,0.386,0.382,0.388,0.395,0.394,0.383,0.387,0.397,
                       0.39,0.385,0.386,0.395,0.399]
#posterior_2levels_AM = [0.387,0.391,0.396,0.394,0.391,0.385,0.39,0.396,0.391,0.385,
                        #0.391,0.391,0.388,0.392,0.389,0.392,0.388,0.392,0.387,0.397,
                        #0.392,0.391,0.388,0.391,0.395]
posterior_2levels_AM = [0.385,0.389,0.388,0.39,0.395,0.381,0.391,0.391,0.395,0.393,0.395,0.387,0.39,0.383,0.392,0.393,0.389,0.388,0.389,0.388,0.389,
                        0.39,0.387,0.389,0.397]


time_ess_one_level = [t / e for t, e in zip(time_one_level, ess_one_level)]
time_ess_two_level = [t / e for t, e in zip(time_two_levels, ess_two_level)]
time_ess_2levels_AM = [t / e for t, e in zip(time_2levels_AM, ess_2levels_AM)]

# Creiamo un DataFrame
df = pd.DataFrame({
    "time_one_level": time_one_level,
    "time_two_levels": time_two_levels,
    "time_2levels_AM":time_2levels_AM,
    "time_per_ESS_one_level": time_ess_one_level,
    "time_per_ESS_two_levels": time_ess_two_level,
    "time_per_ESS_2levels_AM": time_ess_2levels_AM,
    "posterior_one_level": posterior_one_level,
    "posterior_two_levels":posterior_two_level,
    "posterior_2levels_AM":posterior_2levels_AM
})

print(df.head())


plt.figure(figsize=(10, 5))
plt.violinplot([df["time_per_ESS_one_level"], df["time_per_ESS_two_levels"], df["time_per_ESS_2levels_AM"]], showmeans=True)
plt.xticks([1, 2, 3], ["One Level MCMC", "Two Levels MCMC", "Two levels MCMC (AM)"])
plt.ylabel("Time/ESS")
plt.title("sampling efficiency")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.violinplot([df["posterior_one_level"], df["posterior_two_levels"], df["posterior_2levels_AM"]], showmeans=True)
plt.xticks([1, 2, 3], ["One Level MCMC", "Two Levels MCMC", "Two levels MCMC (AM)"])
plt.ylabel("Posterior mean")
plt.title("Posterior identification")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.violinplot([df["time_one_level"], df["time_two_levels"], df["time_2levels_AM"]], showmeans=True)
plt.xticks([1, 2, 3], ["One Level MCMC", "Two Levels MCMC", "Two levels MCMC (AM)"])
plt.ylabel("Time")
plt.title("Time distribution")
plt.grid()
plt.show()

#%%

time_FOM = [216.63,226.97,203.00,217.31,222.73,229.05,231.57,229.12,223.61,223.42,218.72,
                  219.72,225.17,210.65,218.31,224.56,227.34,220.60,220.78,226.91,219.52,228.04,
                  215.17,219.46,222.43]  
time_SGP = [144.13,154.65,142.34,147.36,149.98,158.38,156.02,169.16,161.19,152.05,147.58,137.90,133.51,154.76,
                  153.47,150.32,159.99,156.18,158.17,164.63,133.35,159.94,143.90,154.06,145.63]
time_NN = [159.60,167.09,166.27,175.17,163.37,178.01,165.48,191.75,177.80,171.57,181.22,170.70,165.50,160.40,
                   160.71,172.91,192.40,182.74,191.34,170.07,172.99,185.14,174.61,169.47,178.89]

time_ess_FOM = [5.54,4.61,2.45,4.50,7.19,3.43,4.96,4.90,5.09,3.12,3.28,2.77,3.26,2.10,2.68,5.59,4.91,
                5.84,5.40,4.85,4.25,7.16,4.40,2.81,3.13]
time_ess_SGP = [2.91,4.71,2.82,2.67,4.46,9.94,4.26,6.04,3.82,6.51,5.28,4.50,3.39,4.45,4.85,6.18,8.46,
                 12.57,7.32,5.01,4.68,4.02,2.97,3.31,5.58]
time_ess_NN = [4.29,3.61,4.78,5.79,8.22,4.83,2.91,6.28,6.44,5.87,3.57,4.61,6.03,5.15,4.69,10.84,11.17,
               9.86,11.16,4.39,4.39,5.23,6.64,4.71,10.07]

err_posterior_FOM = [0.644,0.893,0.461,0.481,0.5,0.606,0.530,0.733,0.533,0.783,0.520,0.320,0.386,0.539,
                     0.694,0.589,0.515,0.506,0.837,0.689,0.553,0.726,0.519,0.595,0.448]
err_posterior_SGP = [0.647,0.859,0.370,0.363,0.476,0.629,0.394,0.703,0.481,0.747,0.510,0.347,0.370,0.547,
                     0.705,0.627,0.522,0.573,0.705,0.652,0.486,0.696,0.492,0.571,0.526]
err_posterior_NN = [0.675,0.734,0.507,0.614,0.494,0.589,0.392,0.663,0.529,0.819,0.640,0.320,0.358,0.520,
                    0.766,0.542,0.393,0.473,0.766,0.611,0.611,0.579,0.399,0.482,0.660]

df = pd.DataFrame({
    "time_FOM": time_FOM,
    "time_SGP": time_SGP,
    "time_NN":time_NN,
    "time_ESS_FOM": time_ess_FOM,
    "time_ESS_SGP": time_ess_SGP,
    "time_ESS_NN": time_ess_NN,
    "err_posterior_FOM": err_posterior_FOM,
    "err_posterior_SGP":err_posterior_SGP,
    "err_posterior_NN":err_posterior_NN
})

plt.figure(figsize=(10, 5))
plt.violinplot([df["time_ESS_FOM"], df["time_ESS_SGP"], df["time_ESS_NN"]], showmeans=True)
plt.xticks([1, 2, 3], ["FOM", "SGP", "NN"])
plt.ylabel("Time/ESS")
plt.title("Distribuzione di Time/ESS")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.violinplot([df["err_posterior_FOM"], df["err_posterior_SGP"], df["err_posterior_NN"]], showmeans=True)
plt.xticks([1, 2, 3], ["FOM", "SGP", "NN"])
plt.ylabel("Posterior Samples")
plt.title("Distribuzione di err_posteriors")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.violinplot([df["time_FOM"], df["time_SGP"], df["time_NN"]], showmeans=True)
plt.xticks([1, 2, 3], ["FOM", "SGP", "NN"])
plt.ylabel("Time")
plt.title("Distribuzione di Time")
plt.grid()
plt.show()

#%%

total_samples_list = [2700+300,9000+1000,18000+2000,27000+3000,50000]
rmse = [7.621,5.089,3.762,3.05,2.760]

plt.plot(total_samples_list, rmse, marker='o', label="DNN")
plt.xlabel('Total Samples')
plt.ylabel('RMSE')
#plt.title('Testing RMSE vs Total Samples')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
data_points = [
    (2000, 2.93, -1, "GP full"),       # GP full
    (4000, 2.65, -1, "GP full"),
    
    (10000, 3.388, 400, "Sparse GP"),
    (10000, 2.952, 500, "Sparse GP"),
    (10000, 2.686, 600, "Sparse GP"),
    
    (15000, 2.897, 500, "Sparse GP"),
    (15000, 2.73, 600, "Sparse GP"),
    (15000, 2.41, 700, "Sparse GP"),
]

plt.figure(figsize=(10,6))

# Plot GP full
for N, rmse, ip, label in data_points:
    if ip == -1:
        plt.scatter(N, rmse, color='red', s=80,  marker='x',label='GP full' if 'GP full' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(N, rmse+0.05, "full", color='red', fontsize=9, ha='center')

# Plot Sparse GP
for N, rmse, ip, label in data_points:
    if ip != -1:
        plt.scatter(N, rmse, color='blue', s=80,  marker='x',label='Sparse GP' if 'Sparse GP' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(N, rmse+0.05, f"{ip} IP", color='blue', fontsize=9, ha='center')

plt.xlabel("Total Samples")
plt.ylabel("RMSE")
plt.grid(True)
plt.legend()
plt.ylim(2, 4)
plt.show()


#%%

err_1L = np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_1level_err_001.npy')
time_1L = np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_1level_times_001.npy')

err_DA_SGP =  np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_DA_SGP_err_001.npy')
time_DA_SGP = np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_DA_SGP_times_001.npy')

err_DA_NN =  np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_DA_NN_err_001.npy')
time_DA_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\MDA_MF_DA_NN_times_001.npy')

# Combine data into a single list
data_err = [err_1L, err_DA_SGP, err_DA_NN]
data_times = [time_1L, time_DA_SGP, time_DA_NN]

# Labels for each dataset
labels = ['FOM', 'DA-SGP', 'DA-NN']


# Set the color palette to "crest"
sns.set_palette("BuGn")
#sns.set_palette("Set2")

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=data_err, palette="Set2", alpha=0.9)

# Set title and labels
plt.title('Sampling Error', fontsize=14)
plt.ylabel('Error', fontsize=12)
plt.xticks(ticks=range(3), labels=labels, fontsize=10)
# plt.yscale('log')
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=data_times, palette="Set2", alpha=0.9)

# Set title and labels
plt.title('Sampling Time', fontsize=14)
plt.ylabel('Time(s)', fontsize=12)
plt.xticks(ticks=range(3), labels=labels, fontsize=10)
# plt.yscale('log')
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#%%
x0_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x0_DA.npy')
x1_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_DA.npy')
x2_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x2_DA.npy')
x3_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x3_DA.npy')
x4_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x4_DA.npy')
x5_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x5_DA.npy')
x6_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_DA.npy')
x7_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x7_DA.npy')
x8_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x8_DA.npy')
x9_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x9_DA.npy')
x10_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x10_DA.npy')
x11_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x11_DA.npy')
x12_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x12_DA.npy')
x13_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x13_DA.npy')
x14_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x14_DA.npy')
x15_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_DA.npy')


x0 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x0_FOM.npy')
x1 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_FOM.npy')
x2 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x2_FOM.npy')
x3 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x3_FOM.npy')
x4 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x4_FOM.npy')
x5 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x5_FOM.npy')
x6 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_FOM.npy')
x7 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x7_FOM.npy')
x8 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x8_FOM.npy')
x9 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x9_FOM.npy')
x10 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x10_FOM.npy')
x11 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x11_FOM.npy')
x12 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x12_FOM.npy')
x13 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x13_FOM.npy')
x14 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x14_FOM.npy')
x15 = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_FOM.npy')



x0_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x0_NN.npy')
x1_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_NN.npy')
x2_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x2_NN.npy')
x3_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x3_NN.npy')
x4_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x4_NN.npy')
x5_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x5_NN.npy')
x6_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_NN.npy')
x7_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x7_NN.npy')
x8_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x8_NN.npy')
x9_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x9_NN.npy')
x10_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x10_NN.npy')
x11_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x11_NN.npy')
x12_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x12_NN.npy')
x13_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x13_NN.npy')
x14_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x14_NN.npy')
x15_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_NN.npy')


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns  # per la palette Set2

def plot_trace_and_kde(x_FOM, x_DA, x_NN, theta_label, target_value, param_index):
    methods = ['High-Fidelity', 'Delayed Acceptance (DA)', 'Neural Network (NN)']
    chains = [x_FOM, x_DA, x_NN]  # Ordine corretto: FOM, DA, NN
    legend_labels = ['Markov chain (FOM)', 'Markov chain (DA-SGP)', 'Markov chain (DA-NN)']
    kde_labels = ['Posterior\nKDE', 'Posterior\nKDE', 'Posterior\nKDE']


    set2_colors = sns.color_palette("Set2", 3)
    colors = [set2_colors[0], set2_colors[1], set2_colors[2]]  # Match con l’ordine sopra

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, (chain_raw, method, color) in enumerate(zip(chains, methods, colors)):
        chain = chain_raw.reshape(-1)[:6000]
        mean_val = np.mean(chain)

        # -------- Trace plot (sinistra) --------
        ax = axes[i, 0]
        ax.plot(chain, alpha=0.6, linewidth=0.8, color=color, label=legend_labels[i])
        ax.legend(fontsize=16, loc='upper left', frameon=False)

        for lower, upper, ci_color, alpha_val, label in [
            (2.5, 97.5, 'powderblue', 0.4, '95% CI'),
            (12.5, 87.5, 'lightskyblue', 0.4, '75% CI'),
            (25, 75, 'steelblue', 0.2, '50% CI')
        ]:
            l = np.percentile(chain, lower)
            u = np.percentile(chain, upper)
            ax.fill_between(np.arange(len(chain)), y1=l, y2=u, color=ci_color, alpha=alpha_val, label=label)

        ax.axhline(target_value, color='navy', linewidth=2, label='Target')
        ax.axhline(mean_val, color='darkmagenta', linewidth=2, label='Posterior mean')
        ax.set_ylim(-2.5, 4)
        ax.set_ylabel(fr"$p(\theta_{{{param_index}}} \mid \mathbf{{y}}^{{\mathrm{{exp}}}})$", fontsize=17)

        if i == 2:
            ax.set_xlabel("Discrete step",fontsize=16)
        if i == 0:
            ax.legend(fontsize=16, loc='upper center',ncol=2,frameon=False)
        #ax.set_title(f"Trace Plot — {method}")

        # -------- KDE plot (destra) --------
        ax_kde = axes[i, 1]
        kde = gaussian_kde(chain)
        x_vals = np.linspace(chain.min(), chain.max(), 500)
        y_vals = kde(x_vals)
        ax_kde.plot(x_vals, y_vals, color=color, linewidth=2, label=kde_labels[i])
        
        # Le linee guida restano visive ma non vengono etichettate
        ax_kde.axvline(target_value, color='navy', linewidth=2)
        ax_kde.axvline(mean_val, color='darkmagenta', linewidth=2)
        
        #ax_kde.set_ylabel("Density")
        if i == 2:
            ax_kde.set_xlabel(theta_label,fontsize=16)
        #ax_kde.set_title(f"KDE — {method}")
        
        # Legenda dedicata solo al KDE, con stile e label coerente
        ax_kde.legend(fontsize=16, loc='upper left', frameon=False)

    #fig.suptitle(f"Posterior Analysis for {theta_label}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ------------------------
# Carica i dati
# ------------------------
x1     = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_FOM.npy')
x1_DA  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_DA.npy')
x1_NN  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_NN.npy')

x6     = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_FOM.npy')
x6_DA  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_DA.npy')
x6_NN  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_NN.npy')

x15    = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_FOM.npy')
x15_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_DA.npy')
x15_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_NN.npy')

target_val = [-0.92996394, -0.17701599, -0.31509019,  1.2158185,  -0.53271407, -0.03284841,
              -1.10581973, -1.63113115,  0.28384866,  0.24265321,  0.45266628,  0.54757114,
               0.48490336,  0.63607162, -1.72800687, -0.63881631]

# ------------------------
# Esegui per ogni parametro
# ------------------------
plot_trace_and_kde(x1, x1_DA, x1_NN, r"$\theta_2$", target_val[1],2)
plot_trace_and_kde(x6, x6_DA, x6_NN, r"$\theta_7$", target_val[6],7)
plot_trace_and_kde(x15, x15_DA, x15_NN, r"$\theta_{16}$", target_val[15],16)


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# -------- Caricamento dati --------
x1     = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_FOM.npy')
x1_DA  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_DA.npy')
x1_NN  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x1_NN.npy')

x6     = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_FOM.npy')
x6_DA  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_DA.npy')
x6_NN  = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x6_NN.npy')

x15    = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_FOM.npy')
x15_DA = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_DA.npy')
x15_NN = np.load('C:\\Users\\Ruggi\\Carlotta\\recorded_values\\x15_NN.npy')

# -------- Target values --------
target_val = [-0.92996394, -0.17701599, -0.31509019,  1.2158185,  -0.53271407, -0.03284841,
              -1.10581973, -1.63113115,  0.28384866,  0.24265321,  0.45266628,  0.54757114,
               0.48490336,  0.63607162, -1.72800687, -0.63881631]

# -------- Configurazioni --------
parametri = ['x1', 'x6', 'x15']
target_indices = [1, 6, 15]
dati = [
    [x1, x1_DA, x1_NN],
    [x6, x6_DA, x6_NN],
    [x15, x15_DA, x15_NN]
]
nomi_metodi = ['FOM', 'DA', 'NN']
colori = ['royalblue', 'coral', 'seagreen']

# -------- Trace Plot 3x3 --------
fig_trace, axes_trace = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
for i in range(3):  # righe: parametri
    true_val = target_val[target_indices[i]]
    for j in range(3):  # colonne: metodi
        ax = axes_trace[i, j]
        chain = dati[i][j].reshape(-1)[:6000]
        mean_val = np.mean(chain)
        ax.plot(chain, alpha=0.6, linewidth=0.7, color=colori[j])
        ax.axhline(true_val, color='navy', linestyle='--', linewidth=1.5)
        ax.axhline(mean_val, color='darkmagenta', linestyle='--', linewidth=1.5)
        if i == 2:
            ax.set_xlabel(nomi_metodi[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(parametri[i], fontsize=12)
fig_trace.suptitle("Trace Plot per x1, x6, x15 — Metodi: FOM, DA, NN", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("trace_grid_x1_x6_x15.pdf", dpi=300)

# -------- KDE Plot 3x3 --------
fig_kde, axes_kde = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
for i in range(3):
    true_val = target_val[target_indices[i]]
    for j in range(3):
        ax = axes_kde[i, j]
        chain = dati[i][j].reshape(-1)[:6000]
        mean_val = np.mean(chain)
        kde = gaussian_kde(chain)
        x_vals = np.linspace(chain.min(), chain.max(), 500)
        y_vals = kde(x_vals)
        ax.plot(x_vals, y_vals, color='black', linewidth=2)
        ax.axvline(true_val, color='navy', linestyle='--', linewidth=1.5)
        ax.axvline(mean_val, color='darkmagenta', linestyle='--', linewidth=1.5)
        if i == 2:
            ax.set_xlabel(nomi_metodi[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(parametri[i], fontsize=12)
fig_kde.suptitle("Posterior KDE per x1, x6, x15 — Metodi: FOM, DA, NN", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("kde_grid_x1_x6_x15.pdf", dpi=300)

plt.show()


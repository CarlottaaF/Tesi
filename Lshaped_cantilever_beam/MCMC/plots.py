# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:51:11 2025

@author: Ruggi
"""

import pandas as pd
import matplotlib.pyplot as plt

ess_one_level = [2353,1997,2396,2508,2251,2635,2082,2358,2365,2483,2597,2502,2367,
                 2398,2679,2432,2713,2184,2009,2230,2397,2439,2465,2510,2623]  
ess_two_level = [799.2,524.9,561,609.7,598.2,495,952.7,334.6,607.3,869.9,878.5,518.9,
                 730.1,320.3,1192,1006,708.4,231.9,686.2,946.6,1028,525.8,219.2,
                 896.6,639.4]
time_one_level = [14.57,12.58,12.7,12.69,12.69,12.66,12.56,16.42,16.26,13.63,12.73,
                  12.52,12.52,12.51,12.54,12.51,12.53,12.53,12.54,12.56,12.52,12.53,
                  12.52,12.54,12.53]  
time_two_level = [6.13,6.06,6.79,6.75,6.87,6.84,5.13,5.61,5.12,6,6.02,5.16,6.05,5.61,
                  5.58,6.20,6.05,6.30,6.28,6.19,6.34,6.13,5.68,5.05,5.59]
posterior_one_level = [0.391,0.393,0.389,0.386,0.392,0.391,0.388,0.393,0.389,0.39,
                       0.389,0.391,0.391,0.394,0.39,0.39,0.392,0.393,0.392,0.393,
                       0.392,0.392,0.39,0.391,0.385]
posterior_two_level = [0.39,0.388,0.381,0.389,0.386,0.382,0.388,0.387,0.387,0.393,
                       0.391,0.388,0.386,0.382,0.388,0.395,0.394,0.383,0.387,0.397,
                       0.39,0.385,0.386,0.395,0.399]
time_ess_one_level = [t / e for t, e in zip(time_one_level, ess_one_level)]
time_ess_two_level = [t / e for t, e in zip(time_two_level, ess_two_level)]

# Creiamo un DataFrame
df = pd.DataFrame({
    "time_per_ESS_one_level": time_ess_one_level,
    "time_per_ESS_two_levels": time_ess_two_level,
    "posterior_one_level": posterior_one_level,
    "posterior_two_levels":posterior_two_level
})

print(df.head())


plt.figure(figsize=(10, 5))
plt.violinplot([df["time_per_ESS_one_level"], df["time_per_ESS_two_levels"]], showmeans=True)
plt.xticks([1, 2], ["One Level MCMC", "Two Level MCMC"])
plt.ylabel("Time per ESS")
plt.title("Distribuzione di Time per ESS per metodo")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.violinplot([df["posterior_one_level"], df["posterior_two_levels"]], showmeans=True)
plt.xticks([1, 2], ["One Level MCMC", "Two Level MCMC"])
plt.ylabel("Posterior Samples")
plt.title("Distribuzione dei Posteriori per Metodo")
plt.grid()
plt.show()
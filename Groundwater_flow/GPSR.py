# Import libraries
import GPy as gpy
import numpy as np
from sklearn.cluster import KMeans

np.random.seed(123)

# Import parameters and data
X_train = np.loadtxt('/data_generation_cartella/X_train_h1_16_3.csv', delimiter=',') 
y_train = np.loadtxt('/data_generation_cartella/y_train_h1_16_3.csv', delimiter=',')
print(f"Dimensioni di X_values: {X_train.shape}") #(115200,16)
print(f"Dimensioni di y_values: {y_train.shape}") #(115200,25)

# Add noise to the data

# Da sistemare: prendere solo una parte di questi dati da sporcare e una parte da lasciare pulita
# Ad esempio, 57600 dati sporcati e 57600 dati puliti
# Prenderne meno se il tempo di calcolo è troppo lungo
noise = 0.01
y_observed = y_train + np.random.normal(loc=0.0, scale=noise, size=y_train.shape)
n_samples_train = 115200
datapoints = 25
n_parameters = 16

# Compute the log-likelihood
log_like = np.zeros(n_samples_train)

for i in range(n_samples_train):
    for j in range(datapoints):
        single_value = -0.5 * np.log(2 * np.pi * noise**2) - ((y_observed[i,j] - y_train[i,j]) ** 2) / (2 * noise**2)
        log_like[i] = log_like[i] + single_value

# IMPOSTARE IL KERNEL PER IL GAUSSIAN PROCESS
# classRBF(input_dim, variance=1.0, lengthscale=None, ARD=False, active_dims=None, name='rbf', useGPU=False, inv_l=False
kernel = gpy.kern.RBF(input_dim=n_parameters, variance=1.0, lengthscale=None)


# SCEGLIERE GLI INDUCING POINTS PER LA SPARSE REGRESSION
n_inducing_points = 900

# Casualmente: Seleziona 10 indici casuali da n_samples_train senza ripetizioni
Z = X_train[np.random.choice(n_samples_train, size=n_inducing_points, replace=False), :]

# (Cercare di farlo in modo che siano ben distribuiti sulla superficie di input: maximizing min distance)

# K-means clustering: centroidi dei cluster come inducing points
# **K-Means per selezionare gli inducing points**
# sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=10)
kmeans.fit(X_train)
Z = kmeans.cluster_centers_  # Centroidi dei cluster

# (Ottimizzazione Bayesiana)


# ALLENARE IL MODELLO SRGP

# Creazione del modello di regressione GP sparso
m = gpy.models.SparseGPRegression(X_train, y_train, Z=Z)
m.likelihood.variance = noise**2
print(m)

# **Ottimizzazione 1: fissiamo gli inducing points e ottimizziamo i parametri del kernel**
m.inducing_inputs.fix()
m.optimize('bfgs')
print(m)

# **Ottimizzazione 2: sblocchiamo gli inducing points e riottimizziamo**
m.randomize()
m.Z.unconstrain()
m.optimize('bfgs')
print(m)

# SALVARE IL MODELLO

# FARE PREDIZIONE SUL TEST_SET
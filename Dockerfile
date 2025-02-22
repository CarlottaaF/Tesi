# Usa una base Ubuntu 20.04
FROM ubuntu:20.04

# Imposta variabili d'ambiente per evitare prompt interattivi
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome 

# Installa dipendenze di base e g++-11 per aggiornare libstdc++
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    sudo \
    software-properties-common \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Aggiungi il repository Ubuntu Toolchain per g++-11
RUN sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && sudo apt-get update \
    && sudo apt-get install -y g++-11

# Imposta g++-11 come compilatore di default
RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Scarica e installa Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Imposta il PATH per usare Conda
ENV PATH="/opt/conda/bin:$PATH"

# Inizializza Conda per la shell bash
RUN conda init bash

# Crea un nuovo ambiente Conda con Python 3.10 e installa Fenics e altre librerie
RUN conda update -n base -c defaults conda -y 
RUN conda create -n fenics_env python=3.10 -y
RUN conda install -n fenics_env -c conda-forge fenics numpy scipy matplotlib pyDOE scikit-learn tqdm -y
RUN conda install -n fenics_env -c conda-forge arviz numba -y

RUN /opt/conda/envs/fenics_env/bin/pip install tensorflow keras
RUN /opt/conda/envs/fenics_env/bin/pip install tinyda ray

# Puoi installare GPy, se necessario
RUN /opt/conda/envs/fenics_env/bin/pip install GPy

# Aggiungi il comando per attivare l'ambiente Conda all'avvio del container
# Attiva Conda all'interno del CMD
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate fenics_env && exec bash"]











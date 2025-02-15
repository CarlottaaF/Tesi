# Usa una base Ubuntu 20.04
FROM ubuntu:20.04

# Imposta variabili d'ambiente per evitare prompt interattivi
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome 

# Installa dipendenze di base
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    sudo

# Scarica e installa Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Imposta il PATH per usare Conda
ENV PATH="/opt/conda/bin:$PATH"

# Inizializza Conda per la shell bash
RUN conda init bash

# Crea un nuovo ambiente Conda con Python 3.10 e installa Fenics e altre librerie
RUN conda update -n base -c defaults conda -y && \
    conda create -n fenics_env python=3.10 -y && \
    conda install -n fenics_env -c conda-forge fenics numpy scipy matplotlib pyDOE scikit-learn tqdm -y

# Copia la cartella "data_generation_cartella" all'interno del contenitore
COPY data_generation_cartella /data_generation_cartella

# Aggiungi il comando per attivare l'ambiente Conda all'avvio del container
# Attiva Conda all'interno del CMD
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate fenics_env && exec bash"]








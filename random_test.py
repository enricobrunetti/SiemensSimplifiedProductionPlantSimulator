import numpy as np
import random
import json

import math

import numpy as np
import matplotlib.pyplot as plt

# Dati dei reward per ogni episodio (esempio casuale)
reward_per_episodio = np.random.rand(1000)  # sostituisci con i tuoi dati reali

# Riduci il numero di episodi prendendo solo ogni n-esimo episodio
n = 10  # prendi un episodio ogni 10
reward_per_episodio_downsampled = reward_per_episodio[::n]

# Definisci il kernel per la convoluzione (finestra temporale per la media mobile)
kernel_size = 10  # dimensione della finestra temporale
kernel = np.ones(kernel_size) / kernel_size

# Applica la convoluzione per ottenere la media mobile dei reward
reward_smoothed = np.convolve(reward_per_episodio_downsampled, kernel, mode='valid')

# Plotta i risultati
plt.figure(figsize=(10, 6))
plt.plot(reward_per_episodio_downsampled, label='Reward per episodio (dati originali)', alpha=0.5)
plt.plot(reward_smoothed, label='Reward per episodio (media mobile)', color='red')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.title('Reward per episodio con media mobile (con downsampling)')
plt.legend()
plt.grid(True)
plt.show()


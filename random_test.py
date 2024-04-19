import numpy as np
import random
import json

import math

import numpy as np
import matplotlib.pyplot as plt

# Curve del limite superiore e inferiore dell'intervallo di confidenza (esempio casuale)
upper_bound = np.random.rand(1000) + 1  # limite superiore
lower_bound = np.random.rand(1000)  # limite inferiore

# Riduci il numero di punti prendendo solo ogni n-esimo punto
n = 10  # prendi un punto ogni 10
upper_bound_downsampled = upper_bound[::n]
lower_bound_downsampled = lower_bound[::n]

# Combina i limiti superiore e inferiore per ottenere l'intervallo di confidenza
confidence_interval = upper_bound_downsampled - lower_bound_downsampled

# Definisci il kernel per la convoluzione (finestra temporale per la media mobile)
kernel_size = 10  # dimensione della finestra temporale
kernel = np.ones(kernel_size) / kernel_size

# Applica la convoluzione all'intervallo di confidenza
confidence_interval_smoothed = np.convolve(confidence_interval, kernel, mode='valid')

# Plotta i risultati
plt.figure(figsize=(10, 6))
plt.plot(confidence_interval, label='Intervallo di confidenza (dati originali)', alpha=0.5)
plt.plot(confidence_interval_smoothed, label='Intervallo di confidenza (media mobile)', color='red')
plt.xlabel('Episodio')
plt.ylabel('Intervallo di confidenza')
plt.title('Intervallo di confidenza (con media mobile)')
plt.legend()
plt.grid(True)
plt.show()


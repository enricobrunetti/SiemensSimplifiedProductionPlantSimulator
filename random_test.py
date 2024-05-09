import numpy as np
import random
import json
from utils.graphs_utils import DistQAndLPIPlotter, FQIPlotter
import math

import numpy as np

import numpy as np

def trova_prodotto_agente(matrice_stato, indice_agente):
    riga_agente = matrice_stato[indice_agente]
    indice_prodotto = np.where(riga_agente == 1)[0]
    if len(indice_prodotto) > 0:
        return indice_prodotto[0]
    else:
        return None

# Esempio di utilizzo
stato = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

indice_agente = 2
indice_prodotto = np.where(stato[indice_agente] == 1)[0][0]
print(indice_prodotto)



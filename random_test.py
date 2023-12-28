import numpy as np
import random
import json

# Esempio di liste di liste
lista_principale = [[1, 0, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]]
maschera = [[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]

# Filtraggio delle liste con tutti gli zeri nella maschera
risultato = [lst for lst, mask in zip(lista_principale, maschera) if any(x != 0 for x in mask)]

print(risultato)
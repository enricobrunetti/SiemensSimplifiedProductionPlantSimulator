import numpy as np
import random
                  
'''products_state = np.array([[[1, 2], [3], [4], [5]],
                            [[1], [0], [2], [3]],
                            [[1], [3], [2], [4]],
                            [[1], [0], [0], [2]]])


print(products_state)
products_state[0] = np.maximum(0, products_state[0] - 1)
print(products_state)'''


# Definisci l'array products_state
products_state = np.array([[[1, 2], [3, 0], [4, 0], [5, 0]],
                            [[1, 0], [0, 0], [2, 0], [3, 0]],
                            [[1, 0], [3, 0], [2, 0], [4, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]]])

# Sottrai 1 a tutti gli elementi della prima lista ([[1, 2], [3], [4], [5]])


# Visualizza il risultato
print(np.array(products_state[3]).flatten())
print(all(elem == 0 for elem in np.array(products_state[3]).flatten()))

products_state = np.array([[[0, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 1], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]]])
print(np.where(1 in products_state[2][i] for i in products_state[2]))
print([i for i in range(len(products_state[2])) if 1 in products_state[2][i]][0])


agents_state = np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])
skill = np.argmax(agents_state[0] == 1)
print(skill)
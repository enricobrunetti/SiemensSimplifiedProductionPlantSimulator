import numpy as np
import random
import json

allowed_actions = [6, 7, 10]
prob_values = [0.6, 0.3, 0.1]
indices = [0, 1, 2]
for i in range(100):
    print(np.random.choice(allowed_actions, p=prob_values))








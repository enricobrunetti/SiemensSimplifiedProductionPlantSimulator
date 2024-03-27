import numpy as np
import random
import json

import math

initial_exploration_prob = 1.0
min_exploration_prob = 0.1
decay_rate = 0.95

num_episodes_to_reach_min_prob = math.ceil(math.log(min_exploration_prob / initial_exploration_prob) / math.log(decay_rate))

print(f"Numero di episodi per raggiungere min_exploration_prob: {num_episodes_to_reach_min_prob}")










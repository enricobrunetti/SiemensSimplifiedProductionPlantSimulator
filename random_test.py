import numpy as np
import random
import json
from utils.graphs_utils import DistQAndLPIPlotter, FQIPlotter
import math

import numpy as np

import numpy as np


products_state = [[[1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                    [4, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [5, 0, 0]],

                    [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],

                    [[1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                    [4, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [5, 0, 0]]]

prod = 1

print(all(elem == 0 for elem in np.array(products_state[prod]).flatten()))
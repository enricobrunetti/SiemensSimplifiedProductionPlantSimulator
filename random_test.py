import numpy as np
import random
import json

sorted_list = [{'a': 1, 'b': 20}, {'a': 2, 'b': 15}, {'a': 3, 'b': 10}]

# Trova il dizionario con a == 2
desired_dict = next(item for item in sorted_list if item['a'] == 2)['b']


print(desired_dict)






import numpy as np
import random
import json

CONFIG_PATH = "config/config.json"
                  
with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

agents_connections = {int(k): v for k, v in config['agents_connections'].items()}

print(agents_connections)

agents_state = np.array(config['agents_starting_state'])

print(agents_state)
    
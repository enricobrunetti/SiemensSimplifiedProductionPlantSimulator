import json
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.trajectories_management import extract_agent_trajectories, set_agents_state_observability

#extract_agent_trajectories("prova", 5, 4)

set_agents_state_observability("prova", 5, 4, 1)

'''with open('output/test.json', 'r') as json_file:
    data = json.load(json_file)

mdp = ProductionPlantEnvironment()


chiavi = list(data["Agent: 0"]["Episode 0"][0].keys())

matrice = np.array([[dizionario[chiave] for chiave in chiavi] for dizionario in data["Agent: 0"]["Episode 0"]])
print(matrice)


_, _, _, r, s_prime, absorbing, sa = split_data(matrice, 3, 1)
print(data)'''
import json
import numpy as np
from utils.trajectories_management import split_data_single_agent
from utils.learning_policies_utils import initialize_agents

algorithm = "FQI"
n_episodes = 1000
n_agents = 9
reward_type = "reward5"

learning_agents = initialize_agents(n_agents, algorithm, 5, n_episodes, reward_type, [1,2,3], {})
for agent in learning_agents:
    agent.iter()

_, _, _, r, s_prime, absorbing, sa, _ = split_data_single_agent("output/9_units_siemens_random_test_for_FQI_semiMDP_standard_run0_POSTPROCESSED.json", 2)
example_state = sa[10][:-1]
mask = [1, 1, 1, 1, 1]
action = learning_agents[2].select_action(example_state, mask)
print(f'action selected: {action}')

for agent in learning_agents:
    agent.save()
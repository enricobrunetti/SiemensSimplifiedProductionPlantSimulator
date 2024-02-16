from learning_policies.learning_policies import DistributedQLearningAgent
from utils.trajectories_management import split_data_global
from utils.learning_policies_utils import get_agent_state_and_product_skill_observation_DISTQ
import json
import numpy as np

INPUT_DIR = "output/export_trajectories9_POSTPROCESSED.json"
CONFIG_PATH = "config/DistQ_config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

num_agents = config['n_agents']

agents_dist_q = [DistributedQLearningAgent(config) for _ in range(num_agents)]

_, states, actions, rewards, s_prime, _, _, masks, agents = split_data_global(INPUT_DIR)
print(get_agent_state_and_product_skill_observation_DISTQ(states[0]))
for i in range(len(states) - 1):
    agent = agents_dist_q[agents[i]]
    next_agent = agents_dist_q[agents[i + 1]]
    agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ(s_prime[i]))
    agent.update_values(get_agent_state_and_product_skill_observation_DISTQ(states[i]), actions[i], rewards[i], agents_informations)
    agent.policy_improvement()

selected_action = agents_dist_q[agents[0]].select_action(get_agent_state_and_product_skill_observation_DISTQ(states[0]), masks[0])

print(f'action {selected_action} has been choosen for observation {get_agent_state_and_product_skill_observation_DISTQ(states[0])}')

for _ in range(10):
    i = np.random.randint(len(states))
    selected_action = agents_dist_q[agents[i]].select_action(get_agent_state_and_product_skill_observation_DISTQ(states[i]), masks[i])

    print(f'action {selected_action} has been choosen for observation {get_agent_state_and_product_skill_observation_DISTQ(states[i])}')
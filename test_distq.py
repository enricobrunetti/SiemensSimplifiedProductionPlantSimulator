from learning_policies.learning_policies import QLearningAgent, QLearningDistributedAgent
from utils.trajectories_management import split_data_global
from utils.learning_policies_utils import get_agents_informations
import json

INPUT_DIR = "output/export_trajectories7_POSTPROCESSED.json"
CONFIG_PATH = "config/config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

num_agents = 9
num_products = 5
available_actions = [6, 7, 8, 9, 10]

# basic independent q-learning
agents_qlearning = [QLearningAgent(available_actions, num_products) for _ in range(num_agents)]

_, states, actions, rewards, s_prime, _, _, masks, agents = split_data_global(INPUT_DIR)
for i in range(len(states)):
    agent = agents_qlearning[agents[i]]
    agent.update_q_value(states[i], actions[i], rewards[i], s_prime[i])

selected_action = agents_qlearning[agents[0]].select_action(states[0], 0.5, masks[0])

print(f'action {selected_action} has been choosen for state {states[0]}')

# distributed q-learning
agents_qlearning_dist = [QLearningDistributedAgent(available_actions, num_products) for _ in range(num_agents)]
for i in range(len(states)):
    agent = agents_qlearning_dist[agents[i]]
    agents_informations = get_agents_informations(config, agents_qlearning_dist, agents[i], 1, states[i], actions[i])
    agent.update_q_value(states[i], actions[i], rewards[i], s_prime[i], agents_informations)

selected_action = agents_qlearning_dist[agents[0]].select_action(states[0], 0, masks[0])
print(f'action {selected_action} has been choosen for state {states[0]}')
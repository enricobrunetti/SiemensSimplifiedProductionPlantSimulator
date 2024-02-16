from learning_policies.learning_policies import AgentsManager, LPIAgentV2
from utils.trajectories_management import split_data_global
from utils.learning_policies_utils import get_combined_observation_for_LPI, get_policy_improvement_observation_for_LPI
import json
import numpy as np

INPUT_DIR = "output/export_trajectories9_POSTPROCESSED.json"
CONFIG_PATH = "config/LPI_config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

num_agents = config['n_agents']

agents_manager = AgentsManager(config)
agents_LPI = [LPIAgentV2(config, agents_manager, i) for i in range(num_agents)]

for agent in agents_LPI:
    agents_manager.register_agent(agent)

_, states, actions, rewards, s_prime, _, _, masks, agents = split_data_global(INPUT_DIR)
for i in range(len(states) - 1):
    agent = agents_LPI[agents[i]]
    next_agent = agents_LPI[agents[i + 1]]
    observation = agent.generate_observation(states[i], actions[i])
    next_state_observation = agent.generate_observation(states[i])
    agent.update_values(observation, rewards[i], next_agent.agent_num, next_state_observation)

for _ in range(10):
    i = np.random.randint(len(states))
    agent = agents_LPI[agents[i]]
    observation = agent.generate_observation(states[i])
    selected_action = agents_LPI[agents[i]].select_action(observation, 0, masks[i])

    print(f'agent: {agents[i]}, action {selected_action} has been choosen for state {observation} -> {states[i]}')

'''print(states[0])
print(get_combined_observation_for_LPI(states[0], agents_LPI[agents[0]].neighbours_beta, defer))
for i in range(len(states) - 1):
    # policy evaluation
    agent = agents_LPI[agents[i]]
    next_agent = agents_LPI[agents[i + 1]]
    observation = get_combined_observation_for_LPI(states[i], agents_LPI[agents[i]].neighbours_beta, defer)
    next_observation = get_combined_observation_for_LPI(s_prime[i], agents_LPI[agents[i + 1]].neighbours_beta, defer)
    agents_informations = next_agent.get_max_value(next_agent.agent_num, next_observation)
    agent.update_values(observation, actions[i], rewards[i], agents_informations)
    # policy improvement
    for j in agent.neighbours_kappa:
        policy_improvement_observation = get_policy_improvement_observation_for_LPI(s_prime[i], agent.agent_num, actions[i], config['available_actions'])
        agents_LPI[j].one_policy_improvement(policy_improvement_observation)

'''


'''for episode in data:
    for i in range(len(episode['states']) - 1):
        agent = agents_LPI[episode['agents'][i]]
        next_agent = agents_LPI[episode['agents'][i + 1]]
        # Capire cos'è next_combined_observation che usano loro
        agents_informations = next_agent.get_max_value(episode['agents'][i], get_agent_state_and_product_skill_observation(episode['s_prime'][i]))
        agent.update_values(get_agent_state_and_product_skill_observation(episode['states'][i]), episode['actions'][i], episode['rewards'][i], agents_informations)

for _ in range(10):
    i = np.random.randint(len(data[0]['states']))
    selected_action = agents_LPI[data[0]['agents'][i]].select_action(get_agent_state_and_product_skill_observation(data[0]['states'][i]), 0, data[0]['masks'][i])

    print(f'action {selected_action} has been choosen for state {get_agent_state_and_product_skill_observation(data[0]["states"][i])} -> {data[0]["states"][i]}')

_, states, actions, rewards, s_prime, _, _, masks, agents = split_data_global(INPUT_DIR)
for i in range(len(states)):
    agent = agents_dist_q[agents[i]]
    next_agent = agents_dist_q[agents[i]]
    agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation(s_prime[i]))
    agent.update_values(get_agent_state_and_product_skill_observation(states[i]), actions[i], rewards[i], agents_informations)

selected_action = agents_dist_q[agents[0]].select_action(get_agent_state_and_product_skill_observation(states[0]), 0, masks[0])

print(f'action {selected_action} has been choosen for state {get_agent_state_and_product_skill_observation(states[0])} -> {states[0]}')

for _ in range(10):
    i = np.random.randint(len(states))
    selected_action = agents_dist_q[agents[i]].select_action(get_agent_state_and_product_skill_observation(states[i]), 0, masks[i])

    print(f'action {selected_action} has been choosen for state {get_agent_state_and_product_skill_observation(states[i])} -> {states[i]}')
'''
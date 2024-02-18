from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.learning_policies_utils import initialize_agents, get_agent_state_and_product_skill_observation_DISTQ_online, get_next_agent_number
import json
import numpy as np
import copy

CONFIG_PATH = "config/simulator_config.json"
OUTPUT_PATH = "output/outputDistQTest"
TRAJECTORY_PATH = "output/export_trajectories_distq_test"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

n_agents = config['n_agents']
n_products = config['n_products']
n_episodes = config['n_episodes']
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
nothing_action = n_production_skills + 5
algorithm = config['algorithm']

if algorithm != 'random':
    learning_agents = initialize_agents(n_agents, algorithm)

env = ProductionPlantEnvironment(config)

for episode in range(n_episodes):
    # create log
    with open(f"{OUTPUT_PATH}_{episode}.txt", 'w') as file:
        file.write('')

    # create trajectory
    trajectories = {f'Episode {episode}, Product {product}': [] for product in range(n_products)}
    with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
        json.dump(trajectories, outfile, indent=6)

    state = env.reset()
    old_state = copy.deepcopy(state)

    for step in range(num_max_steps):
        with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
            file.write(f"***************Episode{episode}***************\n")
            file.write(f"***************Step{step}***************\n")
            file.write(f"Time: {state['time']}, Current agent: {state['current_agent']}\n")
            file.write(f"Agents busy: {state['agents_busy']}\n")
            file.write(f"Agents state: \n{state['agents_state']}\n")
            file.write(f"Products state: \n{state['products_state']}\n")
            file.write(f"Action mask: \n{state['action_mask']}\n")

        action_selected_by_algorithm = False

        if np.all(state['action_mask'][state['current_agent']] == 0) or state['agents_busy'][state['current_agent']][0] == 1:
            # if no actions available -> do nothing
            action = nothing_action
        else:
            actions = np.array(env.actions)
            actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
            # if only production actions are available randomly select one of them
            if np.max(actions) < n_production_skills:
                action = np.random.choice(actions)
            # otherwise select the proper transfer action with the selected algorithm
            else:
                action_selected_by_algorithm = True
                if algorithm == 'random':
                    action = np.random.choice(actions)
                elif algorithm == 'DistQ':
                    agent = learning_agents[state['current_agent']]
                    obs = get_agent_state_and_product_skill_observation_DISTQ_online(state['current_agent'], state)
                    mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                    action = agent.select_action(obs, mask)

        state, reward, done, _ = env.step(action)

        if action_selected_by_algorithm:
            if algorithm == 'DistQ':
                agent_num = old_state['current_agent']
                agent = learning_agents[agent_num]
                next_agent_num = get_next_agent_number(config, agent_num, action)
                next_agent = learning_agents[next_agent_num]

                agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ_online(next_agent_num, state))
                obs = get_agent_state_and_product_skill_observation_DISTQ_online(agent_num, old_state)
                agent.update_values(obs, action, reward, agents_informations)
                agent.policy_improvement()

        with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
            file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

        # update trajectory
        if action != nothing_action:
            with open(f"{TRAJECTORY_PATH}_{episode}.json", 'r') as infile:
                trajectories = json.load(infile)
            
            state_to_save = copy.deepcopy(old_state)
            del state_to_save['time'], state_to_save['current_agent'], state_to_save['agents_busy']
            state_to_save['action_mask'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save['action_mask'].items()}
            state_to_save = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save.items()}
            trajectory_update = {'time': int(old_state['time']), 'agent': int(old_state['current_agent']), 'state': state_to_save, 'action': int(action), 'reward': int(reward)}
            current_product = np.argmax(state['agents_state'][old_state['current_agent']]) if action == 0 else np.argmax(old_state['agents_state'][old_state['current_agent']])
            trajectories[f"Episode {episode}, Product {current_product}"].append(trajectory_update)
            
            with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
                json.dump(trajectories, outfile, indent=6)
        
        old_state = copy.deepcopy(state)

        if done:
            print(f"The episode {episode+1}/{n_episodes} is finished.")
            break
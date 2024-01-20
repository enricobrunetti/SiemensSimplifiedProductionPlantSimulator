import json
import numpy as np
import copy
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment

CONFIG_PATH = "config/config.json"
OUTPUT_PATH = "output/output6"
TRAJECTORY_PATH = "output/export_trajectories6"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

n_products = config['n_products']
n_runs = config['n_runs']
num_max_steps = config['num_max_steps']
nothing_action = config['n_production_skills'] + 5

env = ProductionPlantEnvironment(config)

for run in range(n_runs):
    # create log
    with open(f"{OUTPUT_PATH}_{run}.txt", 'w') as file:
        file.write('')

    # create trajectory
    trajectories = {f'Episode {episode}': [] for episode in range(n_products * run, n_products * run + n_products)}
    with open(f"{TRAJECTORY_PATH}_{run}.json", 'w') as outfile:
        json.dump(trajectories, outfile, indent=6)

    state = env.reset()
    old_state = copy.deepcopy(state)

    for step in range(num_max_steps):
        with open(f"{OUTPUT_PATH}_{run}.txt", 'a') as file:
            file.write(f"***************Run{run}***************\n")
            file.write(f"***************Step{step}***************\n")
            file.write(f"Time: {state['time']}, Current agent: {state['current_agent']}\n")
            file.write(f"Agents busy: {state['agents_busy']}\n")
            file.write(f"Agents state: \n{state['agents_state']}\n")
            file.write(f"Products state: \n{state['products_state']}\n")
            file.write(f"Action mask: \n{state['action_mask']}\n")

        if np.all(state['action_mask'][state['current_agent']] == 0) or state['agents_busy'][state['current_agent']][0] == 1:
            # if no actions available -> do nothing
            action = nothing_action
        else:
            actions = np.array(env.action_space)
            actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
            action = np.random.choice(actions)

        state, reward, done, _ = env.step(action)

        with open(f"{OUTPUT_PATH}_{run}.txt", 'a') as file:
            file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

        # update trajectory
        if action != nothing_action:
            with open(f"{TRAJECTORY_PATH}_{run}.json", 'r') as infile:
                trajectories = json.load(infile)
            
            state_to_save = copy.deepcopy(old_state)
            del state_to_save['time'], state_to_save['current_agent'], state_to_save['agents_busy']
            state_to_save['action_mask'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save['action_mask'].items()}
            state_to_save = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save.items()}
            trajectory_update = {'time': int(old_state['time']), 'agent': int(old_state['current_agent']), 'state': state_to_save, 'action': int(action), 'reward': int(reward)}
            current_product = np.argmax(state['agents_state'][old_state['current_agent']]) if action == 0 else np.argmax(old_state['agents_state'][old_state['current_agent']])
            trajectories[f"Episode {current_product + n_products * run}"].append(trajectory_update)
            
            with open(f"{TRAJECTORY_PATH}_{run}.json", 'w') as outfile:
                json.dump(trajectories, outfile, indent=6)
        
        old_state = copy.deepcopy(state)

        if done:
            print(f"The run {run+1}/{n_runs} is finished.")
            break
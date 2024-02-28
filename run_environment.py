from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.learning_policies_utils import initialize_agents, get_agent_state_and_product_skill_observation_DISTQ_online
from utils.graphs_utils import RewardVisualizer
import json
import numpy as np
import copy

CONFIG_PATH = "config/simulator_config.json"
OUTPUT_PATH = "output/outputDistQTest"
TRAJECTORY_PATH = "output/export_trajectories_distq_test"
LOG_PATH = "output/run_log_LPI_softmax"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

n_agents = config['n_agents']
n_products = config['n_products']
n_episodes = config['n_episodes']
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
nothing_action = n_production_skills + 5
algorithm = config['algorithm']
export_trajectories = config['export_trajectories']
custom_reward = config['custom_reward']

with open(f"{LOG_PATH}.txt", 'w') as file:
    file.write(f"")

if algorithm != 'random':
    learning_agents, policy_improvement = initialize_agents(n_agents, algorithm)

if algorithm == 'LPI':
    observations_history_LPI = {}

env = ProductionPlantEnvironment(config)
reward_visualizer = RewardVisualizer(n_agents)

performance = {}

for episode in range(n_episodes):
    # create log
    with open(f"{OUTPUT_PATH}_{episode}.txt", 'w') as file:
        file.write('')

    if export_trajectories:
        # create trajectory
        trajectories = {f'Episode {episode}, Product {product}': [] for product in range(n_products)}
        with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
            json.dump(trajectories, outfile, indent=6)

    with open(f"{LOG_PATH}.txt", 'a') as file:
        file.write(f"Starting episode {episode+1}\n")

    agents_rewards = [[] for _ in range(n_agents)]
    if custom_reward == 'negative':
        agents_seen_products = [[0 for i in range(n_products)] for j in range(n_agents)]

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
            # if only production actions are available there will be only one element
            if np.max(actions) < n_production_skills:
                action = actions[0]

                with open(f"{LOG_PATH}.txt", 'a') as file:
                    if old_state['current_agent'] == 0:
                        file.write(f"New product picked up by agent {old_state['current_agent']}\n")
                    else:
                        product = np.argmax(old_state['agents_state'][old_state['current_agent']])
                        file.write(f"Production skill {action} performed on product {product} by agent {old_state['current_agent']}\n")

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
                elif algorithm == 'LPI':
                    agent = learning_agents[state['current_agent']]
                    obs = agent.generate_observation(state)
                    mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                    action = agent.select_action(obs, mask)


        state, reward, done, _ = env.step(action)

        if action_selected_by_algorithm:
            agent_num = old_state['current_agent']

            if custom_reward == 'negative':
                if 1 in old_state['agents_state'][agent_num]:
                        product = np.where(old_state['agents_state'][agent_num] == 1)[0][0]
                        agents_seen_products[agent_num][product] = 1

                for j in range(n_products):
                    if all(elem == 0 for elem in np.array(state['products_state'][j]).flatten()):
                        agents_seen_products[agent_num][j] = 0
                
                reward = -1 * agents_seen_products[agent_num].count(1)

            agents_rewards[agent_num].append(reward)

            agent = learning_agents[agent_num]
            if algorithm == 'DistQ':
                next_agent_num = agent.get_next_agent_number(action)
                next_agent = learning_agents[next_agent_num]

                agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ_online(next_agent_num, state))
                obs = get_agent_state_and_product_skill_observation_DISTQ_online(agent_num, old_state)
                agent.update_values(obs, action, reward, agents_informations)
            elif algorithm == 'LPI':
                if agent_num not in observations_history_LPI.keys():
                    observations_history_LPI[agent_num] = {}
                    observations_history_LPI[agent_num]['O'] = []
                    observations_history_LPI[agent_num]['A'] = []
                    observations_history_LPI[agent_num]['R'] = []

                observations_history_LPI[agent_num]['O'].append(agent.generate_observation(old_state))
                observations_history_LPI[agent_num]['A'].append(action)
                observations_history_LPI[agent_num]['R'].append(reward)

                if len(observations_history_LPI[agent_num]['O']) > 1:
                    agent.update_values(observations_history_LPI[agent_num]['O'][-2], \
                                        observations_history_LPI[agent_num]['A'][-2], \
                                        observations_history_LPI[agent_num]['R'][-2], \
                                        observations_history_LPI[agent_num]['O'][-1], \
                                        observations_history_LPI[agent_num]['A'][-1])
            
            if policy_improvement == 'step':
                agent.soft_policy_improvement()
            
            with open(f"{LOG_PATH}.txt", 'a') as file:
                source_agent = old_state['current_agent']
                product = np.argmax(old_state['agents_state'][old_state['current_agent']])
                for i, sublist in enumerate(state['agents_state']):
                    if sublist[product] == 1:
                        dest_agent = i
                        break
                file.write(f"Agent {source_agent} transfer product {product} to agent {dest_agent} (action: {action})\n")
                

        with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
            file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

        if export_trajectories:
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
    
    if policy_improvement == 'episode':
        for agent in learning_agents:
            agent.soft_policy_improvement()

    with open(f"{LOG_PATH}.txt", 'a') as file:
        file.write(f"Episode {episode+1} ended\n")

    performance[episode] = {}
    performance[episode]['episode_duration'] = state['time']

    for i in range(n_agents):
        mean_reward = np.mean(agents_rewards[i])
        reward_visualizer.update_plot(i, episode, mean_reward)

        performance[episode][i] = {}
        performance[episode][i]['mean_reward'] = mean_reward

reward_visualizer.show_plot()

for agent in learning_agents:
    # TO-DO: check if it is possible to pass model_path only one time (since it's the same for all agents)
    model_path = agent.save(n_episodes)

with open(f"{model_path}/training_performance.txt", 'w') as file:
    for i in range(n_episodes):
        file.write(f"****Episode {i+1}****\n")
        file.write(f"Mean reward:\n")
        for j in range(n_agents):
            file.write(f"Agent {j}: {performance[i][j]['mean_reward']}\n")
        file.write(f"Time to complete: {performance[i]['episode_duration']}\n\n")
        
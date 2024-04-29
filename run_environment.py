from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.learning_policies_utils import initialize_agents, get_agent_state_and_product_skill_observation_DISTQ_online
from utils.fqi_utils import get_FQI_state
from utils.graphs_utils import DistQAndLPIPlotter, FQIPlotter
from utils.run_environment_utils import get_next_agent_number
import json
import numpy as np
import copy
import os

CONFIG_PATH = "config/simulator_config.json"
SEMI_MDP_CONFIG_PATH = "config/semiMDP_reward_config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

test_model = config['test_model']
test_model_name = config['test_model_name']
test_model_n_episodes = config['test_model_n_episodes']
n_agents = config['n_agents']
n_products = config['n_products']
n_runs = config['n_runs']
n_episodes = config['n_episodes']
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
nothing_action = n_production_skills + 5
defer_action = n_production_skills + 4
algorithm = config['algorithm']
export_trajectories = config['export_trajectories']
output_log = config['output_log']
custom_reward = config['custom_reward']
supply_action = config['supply_action']
discount_factor = config['gamma']
action_time = config['action_time']
agents_skills_custom_duration = {
    int(outer_key): {
        int(inner_key): value 
        for inner_key, value in inner_dict.items()
    } 
    for outer_key, inner_dict in config['agents_skills_custom_duration'].items()
}
available_actions = config['available_actions']
agents_connections = {int(k): v for k, v in config['agents_connections'].items()}
baseline_path = config['baseline_path']

OUTPUT_PATH = config['output_path']
TRAJECTORY_PATH = config['trajectory_path']

if custom_reward == 'reward5':
    with open(SEMI_MDP_CONFIG_PATH) as config_file:
        semiMDP_reward_config = json.load(config_file)

# Dictionary of the paths of all the runs
model_path_runs = {}
baseline_path_runs = {}

for run in range(n_runs):
    if test_model:
        n_episodes = config['n_episodes']

    if algorithm != 'random' and algorithm != 'FQI':
        learning_agents, update_values, policy_improvement = initialize_agents(n_agents, algorithm, run, n_episodes, custom_reward, config['available_actions'], config['agents_connections'])
    elif algorithm == 'FQI':
        learning_agents = initialize_agents(n_agents, algorithm, run, n_episodes, custom_reward, config['available_actions'], config['agents_connections'])

    if algorithm == 'LPI':
        observations_history_LPI = {}

    env = ProductionPlantEnvironment(config)

    performance = {}

    if algorithm != 'random':
        if test_model:
            # if we are testing load existing model
            for i in range(len(learning_agents)):
                learning_agents[i].load(f'{test_model_name}/run{run}/{i}')
        else:
            # otherwise create model path
            for agent in learning_agents:
                agent.save()
        model_path = learning_agents[0].get_model_name()
    else:
        model_path = baseline_path
        model_path += f'/run{run}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    model_path_runs[run] = model_path
    baseline_path_runs[run] = baseline_path
    baseline_path_runs[run] += f'/run{run}'

    if test_model:
        file_log_name = f"{model_path}/test_logs.txt"
    else:
        file_log_name = f"{model_path}/training_logs.txt"
    with open(file_log_name, 'w') as file:
        file.write(f"")

    if test_model:
        n_episodes = test_model_n_episodes

    for episode in range(n_episodes):
        if output_log:
            # create log
            with open(f"{OUTPUT_PATH}_run{run}_{episode}.txt", 'w') as file:
                file.write('')

        if export_trajectories:
            # create trajectory
            trajectories = {f'Episode {episode}, Product {product}': [] for product in range(n_products)}
            with open(f"{TRAJECTORY_PATH}_run{run}_{episode}.json", 'w') as outfile:
                json.dump(trajectories, outfile, indent=6)

        with open(file_log_name, 'a') as file:
            file.write(f"Starting episode {episode+1}\n")

        # used for reward computation for new graph
        agents_rewards_for_plot = {}
        for i in range(n_agents):
            agents_rewards_for_plot[i] = 0

        if custom_reward == 'reward1' or custom_reward == 'reward2' or custom_reward == 'reward4':
            agents_seen_products = [[0 for _ in range(n_products)] for _ in range(n_agents)]
        if custom_reward == 'reward3' or custom_reward == 'reward4':
            n_supplied_products = 0

        if algorithm == 'DistQ' or algorithm == 'LPI':
            for agent in learning_agents:
                if test_model:
                    agent.update_exploration_prob(test_model)
                else:
                    # update exploration probability basing on episode
                    agent.update_exploration_prob(test_model, episode + 1)

        if algorithm == 'FQI' and not test_model:
            for fqi_agent in learning_agents:
                fqi_agent.iter()
                fqi_agent.save()

        if custom_reward == 'reward5':
            trajectory_for_semi_MDP = []

        state = env.reset()
        old_state = copy.deepcopy(state)

        for step in range(num_max_steps):
            if output_log and custom_reward != 'reward5':
                with open(f"{OUTPUT_PATH}_run{run}_{episode}.txt", 'a') as file:
                    file.write(f"***************Episode{episode}***************\n")
                    file.write(f"***************Step{step}***************\n")
                    file.write(f"Time: {state['time']}, Current agent: {state['current_agent']}\n")
                    file.write(f"Agents busy: {state['agents_busy']}\n")
                    file.write(f"Agents state: \n{state['agents_state']}\n")
                    file.write(f"Products state: \n{state['products_state']}\n")
                    file.write(f"Action mask: \n{state['action_mask']}\n")

            action_selected_by_algorithm = False

            if np.all(np.array(state['action_mask'][state['current_agent']]) == 0) or state['agents_busy'][state['current_agent']][0] == 1:
                # if no actions available -> do nothing
                action = nothing_action
            else:
                actions = np.array(env.actions)
                actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                # if only production actions are available there will be only one element
                if np.max(actions) < n_production_skills:
                    action = actions[0]

                    with open(file_log_name, 'a') as file:
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
                    elif algorithm == 'FQI':
                        agent = learning_agents[state['current_agent']]
                        obs = get_FQI_state({'agents_state': state['agents_state'].copy(), 'products_state': state['products_state'].copy()}, agent.get_observable_neighbours(), n_products)
                        mask = state['action_mask'][state['current_agent']][n_production_skills:-1]
                        action = agent.select_action(obs, mask)

            state, reward, done, _ = env.step(action)

            if (custom_reward == 'reward3' or custom_reward == 'reward4') and action == supply_action:
                n_supplied_products += 1

            if custom_reward == 'reward5':
                trajectory_for_semi_MDP.append({'step': step, 'old_state': old_state, 'action': action, 'reward': reward, 'new_state': state, 'action_selected_by_algorithm': action_selected_by_algorithm})

            if action_selected_by_algorithm:
                agent_num = old_state['current_agent']

                if custom_reward == 'reward1' or custom_reward == 'reward4':
                    if 1 in old_state['agents_state'][agent_num]:
                        product = np.where(old_state['agents_state'][agent_num] == 1)[0][0]
                        agents_seen_products[agent_num][product] = 1

                    for j in range(n_products):
                        if all(elem == 0 for elem in np.array(state['products_state'][j]).flatten()):
                            agents_seen_products[agent_num][j] = 0
                    
                    reward = -1 * agents_seen_products[agent_num].count(1)

                elif custom_reward == 'reward2':
                    product = np.where(old_state['agents_state'][agent_num] == 1)[0][0]
                    agents_seen_products[agent_num][product] += 1

                    reward = 100 * agents_seen_products[agent_num][product] * np.power(learning_agents[agent_num].get_gamma(), old_state['time'])

                elif custom_reward == 'reward3':
                    reward = -1 * (n_products - n_supplied_products)

                if custom_reward == 'reward4' and ((n_products - n_supplied_products) != 0):
                    reward = -100

                # check: verifica se è corretto old_state
                if custom_reward != 'reward5':
                    agents_rewards_for_plot[agent_num] += np.power(discount_factor,  old_state['time']) * reward

                if algorithm != 'random':
                    agent = learning_agents[agent_num]
                if not test_model and not custom_reward == 'reward5' and algorithm == 'DistQ':
                    next_agent_num = agent.get_next_agent_number(action)
                    next_agent = learning_agents[next_agent_num]

                    agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ_online(next_agent_num, state))
                    obs = get_agent_state_and_product_skill_observation_DISTQ_online(agent_num, old_state)
                    agent.update_values(obs, action, reward, agents_informations)
                    if update_values == 'step':
                        agent.apply_values_update()

                elif not test_model and not custom_reward == 'reward5' and algorithm == 'LPI':
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
                        if update_values == 'step':
                            agent.apply_values_update()
                
                if algorithm != 'random':
                    if not test_model and not custom_reward == 'reward5' and policy_improvement == 'step':
                        agent.soft_policy_improvement()
                
                with open(file_log_name, 'a') as file:
                    source_agent = old_state['current_agent']
                    product = np.argmax(old_state['agents_state'][old_state['current_agent']])
                    for i, sublist in enumerate(state['agents_state']):
                        if sublist[product] == 1:
                            dest_agent = i
                            break
                    file.write(f"Agent {source_agent} transfer product {product} to agent {dest_agent} (action: {action})\n")
                    
            if output_log and custom_reward != 'reward5':
                with open(f"{OUTPUT_PATH}_run{run}_{episode}.txt", 'a') as file:
                    file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

            if export_trajectories and custom_reward != 'reward5':
                # update trajectory
                if action != nothing_action:
                    with open(f"{TRAJECTORY_PATH}_run{run}_{episode}.json", 'r') as infile:
                        trajectories = json.load(infile)
                    
                    state_to_save = copy.deepcopy(old_state)
                    del state_to_save['time'], state_to_save['current_agent'], state_to_save['agents_busy']
                    state_to_save['action_mask'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save['action_mask'].items()}
                    state_to_save = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save.items()}
                    trajectory_update = {'time': int(old_state['time']), 'agent': int(old_state['current_agent']), 'state': state_to_save, 'action': int(action), 'reward': int(reward)}
                    current_product = np.argmax(state['agents_state'][old_state['current_agent']]) if action == 0 else np.argmax(old_state['agents_state'][old_state['current_agent']])
                    trajectories[f"Episode {episode}, Product {current_product}"].append(trajectory_update)
                    
                    with open(f"{TRAJECTORY_PATH}_run{run}_{episode}.json", 'w') as outfile:
                        json.dump(trajectories, outfile, indent=6)
            
            old_state = copy.deepcopy(state)

            if done:
                print(f"The episode {episode+1}/{n_episodes} is finished.")
                break

        # semi MDP reward postponed computation (at the end of the episode)
        if custom_reward == 'reward5':
            for actual_step in range(len(trajectory_for_semi_MDP) - 1):
                if trajectory_for_semi_MDP[actual_step]['action_selected_by_algorithm']:
                    current_agent = trajectory_for_semi_MDP[actual_step]['old_state']['current_agent']
                    actual_time = trajectory_for_semi_MDP[actual_step]['old_state']['time']
                    actual_action = trajectory_for_semi_MDP[actual_step]['action']
                    next_step = actual_step + 1
                    while next_step < (len(trajectory_for_semi_MDP) - 1) and trajectory_for_semi_MDP[next_step]['old_state']['agents_state'][current_agent][0] != 1:
                        next_step += 1
                    next_time = trajectory_for_semi_MDP[next_step]['old_state']['time']
                    trajectory_for_semi_MDP[actual_step]['reward'] = -1 * (next_time - actual_time)
                    if algorithm != 'random':
                        next_agent = learning_agents[current_agent].get_next_agent_number(actual_action)
                    else:
                        next_agent = get_next_agent_number(current_agent, actual_action, available_actions, agents_connections)
                    if semiMDP_reward_config['positive_shaping']:
                        for check_positive_shaping_step in range(len(trajectory_for_semi_MDP) - actual_step - 1):
                            if trajectory_for_semi_MDP[actual_step + check_positive_shaping_step + 1]['old_state']['current_agent'] == next_agent:
                                next_agent_action = trajectory_for_semi_MDP[actual_step + check_positive_shaping_step + 1]['action']
                                if next_agent_action != nothing_action:
                                    if next_agent_action < n_production_skills:
                                        if next_agent in agents_skills_custom_duration and next_agent_action in agents_skills_custom_duration[next_agent]:
                                            next_agent_action_time = agents_skills_custom_duration[next_agent][next_agent_action]
                                        else:
                                            next_agent_action_time = action_time[next_agent_action]
                                        trajectory_for_semi_MDP[actual_step]['reward'] = next_agent_action_time * semiMDP_reward_config['positive_shaping_constant']
                                    break
                    if actual_action == defer_action and semiMDP_reward_config['negative_shaping']:
                        trajectory_for_semi_MDP[actual_step]['reward'] = -1 * semiMDP_reward_config['negative_shaping_constant']


            # semi MDP reward postponed update values (at the end of the episode)
            if not test_model and custom_reward == 'reward5' and algorithm == 'DistQ':
                for actual_step in trajectory_for_semi_MDP:
                    if actual_step['action_selected_by_algorithm']:
                        agent = learning_agents[actual_step['old_state']['current_agent']]
                        action = actual_step['action']
                        reward = actual_step['reward']
                        next_agent_num = agent.get_next_agent_number(action)
                        next_agent = learning_agents[next_agent_num]

                        agents_informations = next_agent.get_max_value(get_agent_state_and_product_skill_observation_DISTQ_online(next_agent_num, actual_step['new_state']))
                        obs = get_agent_state_and_product_skill_observation_DISTQ_online(agent_num, actual_step['old_state'])
                        agent.update_values(obs, action, reward, agents_informations)

            elif not test_model and custom_reward == 'reward5' and algorithm == 'LPI':
                for actual_step in trajectory_for_semi_MDP:
                    if actual_step['action_selected_by_algorithm']:
                        agent_num = actual_step['old_state']['current_agent']
                        agent = learning_agents[agent_num]
                        action = actual_step['action']
                        reward = actual_step['reward']
                        if agent_num not in observations_history_LPI.keys():
                            observations_history_LPI[agent_num] = {}
                            observations_history_LPI[agent_num]['O'] = []
                            observations_history_LPI[agent_num]['A'] = []
                            observations_history_LPI[agent_num]['R'] = []

                        observations_history_LPI[agent_num]['O'].append(agent.generate_observation(actual_step['old_state']))
                        observations_history_LPI[agent_num]['A'].append(action)
                        observations_history_LPI[agent_num]['R'].append(reward)

                        if len(observations_history_LPI[agent_num]['O']) > 1:
                            agent.update_values(observations_history_LPI[agent_num]['O'][-2], \
                                                observations_history_LPI[agent_num]['A'][-2], \
                                                observations_history_LPI[agent_num]['R'][-2], \
                                                observations_history_LPI[agent_num]['O'][-1], \
                                                observations_history_LPI[agent_num]['A'][-1])
                            
            # semiMDP agents reward for plot
            for actual_step in trajectory_for_semi_MDP:
                if actual_step['action_selected_by_algorithm']:
                    agents_rewards_for_plot[actual_step['old_state']['current_agent']] += np.power(discount_factor,  actual_step['old_state']['time']) * actual_step['reward']
                            
            # semiMDP output log
            if output_log:
                with open(f"{OUTPUT_PATH}_run{run}_{episode}.txt", 'a') as file:
                    for actual_step in trajectory_for_semi_MDP:
                        file.write(f"***************Episode{episode}***************\n")
                        file.write(f"***************Step{actual_step['step']}***************\n")
                        file.write(f"Time: {actual_step['old_state']['time']}, Current agent: {actual_step['old_state']['current_agent']}\n")
                        file.write(f"Agents busy: {actual_step['old_state']['agents_busy']}\n")
                        file.write(f"Agents state: \n{actual_step['old_state']['agents_state']}\n")
                        file.write(f"Products state: \n{actual_step['old_state']['products_state']}\n")
                        file.write(f"Action mask: \n{actual_step['old_state']['action_mask']}\n")
                        file.write(f"Step: {step}, Action: {actual_step['action']}, Reward: {actual_step['reward']}\n\n")

            # semiMDP export trajectories
            if export_trajectories:
                with open(f"{TRAJECTORY_PATH}_run{run}_{episode}.json", 'r') as infile:
                    trajectories = json.load(infile)
                
                for actual_step in trajectory_for_semi_MDP:
                    if actual_step['action'] != nothing_action:
                        state_to_save = copy.deepcopy(actual_step['old_state'])
                        del state_to_save['time'], state_to_save['current_agent'], state_to_save['agents_busy']
                        state_to_save['action_mask'] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save['action_mask'].items()}
                        state_to_save = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in state_to_save.items()}
                        trajectory_update = {'time': int(actual_step['old_state']['time']), 'agent': int(actual_step['old_state']['current_agent']), 'state': state_to_save, 'action': int(actual_step['action']), 'reward': int(actual_step['reward'])}
                        current_product = np.argmax(state['agents_state'][old_state['current_agent']]) if action == 0 else np.argmax(old_state['agents_state'][old_state['current_agent']])
                        trajectories[f"Episode {episode}, Product {current_product}"].append(trajectory_update)
                
                with open(f"{TRAJECTORY_PATH}_run{run}_{episode}.json", 'w') as outfile:
                    json.dump(trajectories, outfile, indent=6)

        if algorithm != 'random' and algorithm != 'FQI':
            if not test_model and update_values == 'episode':
                for agent in learning_agents:
                    agent.apply_values_update()
            
            if not test_model and policy_improvement == 'episode':
                for agent in learning_agents:
                    agent.soft_policy_improvement()

        with open(file_log_name, 'a') as file:
            file.write(f"Episode {episode+1} ended\n")

        performance[episode] = {}
        performance[episode]['episode_duration'] = state['time']
        performance[episode]['agents_reward_for_plot'] = agents_rewards_for_plot

        for i in range(n_agents):
            mean_reward = np.mean(agents_rewards_for_plot[i])

            performance[episode][i] = {}
            performance[episode][i]['mean_reward'] = mean_reward

    if test_model:
        plot_path = f"{model_path}/test_reward_graph.png"
        info_for_plot_path = f"{model_path}/reward_for_plot_test.json"
    else:
        plot_path = f"{model_path}/training_reward_graph.png"
        info_for_plot_path = f"{model_path}/reward_for_plot_training.json"

    with open(info_for_plot_path, 'w') as outfile:
        json.dump(performance, outfile, indent=6)

    if not test_model and algorithm != 'random' and algorithm != 'FQI':
        for agent in learning_agents:
            agent.save()

    if test_model:
        performance_log_name = f"{model_path}/test_performance.txt"
    else:
        performance_log_name = f"{model_path}/training_performance.txt"

    with open(performance_log_name, 'w') as file:
        episodes_time_to_complete = [performance[i]['episode_duration'] for i in range(n_episodes)]
        file.write(f"Min time to complete: {np.min(episodes_time_to_complete)}\n")
        file.write(f"Avg time to complete: {np.average(episodes_time_to_complete)}\n")
        file.write(f"Max time to complete: {np.max(episodes_time_to_complete)}\n")
        file.write(f"Variance of time to complete: {np.var(episodes_time_to_complete)}\n")
        for j in range(n_agents):
            file.write(f"Agent {j} mean reward: {np.nanmean([performance[i][j]['mean_reward'] for i in range(n_episodes)])}\n")
        file.write(f"\n")
        for i in range(n_episodes):
            file.write(f"****Episode {i+1}****\n")
            file.write(f"Mean reward:\n")
            for j in range(n_agents):
                file.write(f"Agent {j}: {performance[i][j]['mean_reward']}\n")
            file.write(f"Time to complete: {performance[i]['episode_duration']}\n\n")

if test_model:
    if algorithm != 'random' and algorithm != 'FQI':
        plotter = DistQAndLPIPlotter(model_path_runs, baseline_path_runs)
        plotter.plot_reward_graphs()
        plotter.plot_performance_graph()

if algorithm == 'FQI' and not test_model:
    plotter = FQIPlotter(model_path_runs)
    plotter.plot_reward_graphs()
    plotter.plot_performance_graph()
        
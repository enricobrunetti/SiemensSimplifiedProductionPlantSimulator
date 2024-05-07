from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
import sys
sys.path.insert(0, 'ai_optimizer')
from utils.learning_policies_utils import initialize_agents, get_agent_state_and_product_skill_observation_DISTQ_online
from utils.fqi_utils import get_FQI_state
from utils.graphs_utils import DistQAndLPIPlotter, RewardVisualizer
import json
import numpy as np
import copy



CONFIG_PATH = "config/simulator_config_3units.json"
SERVER_BASE_PORT = 9900
MQTT_HOST_URL = 'tcp://127.0.0.1:1883'
with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

test_model = config['test_model']
test_model_name = config['test_model_name']
test_model_n_episodes = config['test_model_n_episodes']
n_agents = config['n_agents']
n_products = config['n_products']
n_episodes = config['n_episodes']
num_max_steps = config['num_max_steps']
n_production_skills = config['n_production_skills']
nothing_action = n_production_skills + 5
algorithm = config['algorithm']
export_trajectories = config['export_trajectories']
output_log = config['output_log']
custom_reward = config['custom_reward']
supply_action = config['supply_action']
discount_factor = config['gamma']
baseline_path = config['baseline_path']
one_hot_state = config['one_hot_state']
loop_threshold = config["loop_threshold"]
checkpoint_frequency = config["checkpoint_frequency"]


def get_rllib_state(state, old_state, one_hot_state=False, threshold=500):
    # next_skill , previous_agent, threshold_detected
    obs_rllib = []
    next_skill = state["product_state"][:, 0].tolist().index(1)
    previous_agent = old_state["current_agent"]
    if one_hot_state:
        next_skill_ohe = np.zeros(n_production_skills)
        next_skill_ohe[next_skill] = 1
        next_skill = next_skill_ohe
        previous_agent_ohe = np.zeros(n_agents)
        previous_agent_ohe[previous_agent] = 1
        previous_agent = previous_agent_ohe
    threshold_detected = state["time"] > threshold  # TODO: implement a  real threshold check
    obs_rllib.extend(next_skill)
    obs_rllib.extend(previous_agent)
    obs_rllib.extend([threshold_detected])
    return np.array(obs_rllib).flatten()


def convert_action(action):
    return action

def get_cppu_name(agent):
    return f'cppu_{agent}'

OUTPUT_PATH = config['output_path']
TRAJECTORY_PATH = config['trajectory_path']
communicator = initialize_agents(n_agents, algorithm, n_episodes, custom_reward,
                          config['available_actions'], config['agents_connections'], mqtt_host_url=MQTT_HOST_URL)
env = ProductionPlantEnvironment(config)
reward_visualizer = RewardVisualizer(n_agents)
performance = {}
model_path = "models/rllib/"


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
        with open(f"{OUTPUT_PATH}_{episode}.txt", 'w') as file:
            file.write('')

    if export_trajectories:
        # create trajectory
        trajectories = {f'Episode {episode}, Product {product}': [] for product in range(n_products)}
        with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
            json.dump(trajectories, outfile, indent=6)

    with open(file_log_name, 'a') as file:
        file.write(f"Starting episode {episode+1}\n")

    # used for old reward counting (before new graph)
    agents_rewards = [[] for _ in range(n_agents)]
    # used for reward computation for new graph
    agents_rewards_for_plot = {}
    for i in range(n_agents):
        agents_rewards_for_plot[i] = 0
    if custom_reward == 'reward5':
        trajectory_for_semi_MDP = []
    previous_rewards = {}
    state = env.reset()
    old_state = copy.deepcopy(state)
    episode_id = f"E{episode}"
    communicator.publish_episode_management('Start', episode_id)
    communicator.sync_episode()
    for step in range(num_max_steps):
        if output_log and custom_reward != 'reward5':
            with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
                file.write(f"***************Episode{episode}***************\n")
                file.write(f"***************Step{step}***************\n")
                file.write(f"Time: {state['time']}, Current agent: {state['current_agent']}\n")
                file.write(f"Agents busy: {state['agents_busy']}\n")
                file.write(f"Agents state: \n{state['agents_state']}\n")
                file.write(f"Products state: \n{state['products_state']}\n")
                file.write(f"Action mask: \n{state['action_mask']}\n")

        action_selected_by_algorithm = False

        if np.all(np.array(state['action_mask'][state['current_agent']]) == 0) \
                or state['agents_busy'][state['current_agent']][0] == 1:
            # if no actions available -> do nothing
            action = nothing_action
        else:
            actions = np.array(env.actions)
            actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
            # if only production actions are available there will be only one
            # element
            if np.max(actions) < n_production_skills:
                action = actions[0]

                with open(file_log_name, 'a') as file:
                    if old_state['current_agent'] == 0:
                        file.write(f"New product picked up by agent "
                                   f"{old_state['current_agent']}\n")
                    else:
                        product = np.argmax(old_state['agents_state']
                                            [old_state['current_agent']])
                        file.write(f"Production skill {action} performed on"
                                   f" product {product} by agent"
                                   f" {old_state['current_agent']}\n")
            # otherwise select the proper transfer action with
            # the selected algorithm
            else:
                action_selected_by_algorithm = True
                cppu_name = get_cppu_name(state["current_agent"])
                obs = get_rllib_state(state, one_hot_state=one_hot_state, old_state=old_state, threshold=loop_threshold)
                #obs_rllib = []
                communicator.send_state(cppu_name, obs)
                # only works for single product as we need to wait for the action on this agent
                raw_action = communicator.receive_action(cppu_name)
                action = convert_action(raw_action)
        # Here send the message to the workers
        previous_state = state
        state, reward, done, _ = env.step(action)
        if action_selected_by_algorithm:
            previous_rewards[cppu_name] = reward
            next_obs = get_rllib_state(state, one_hot_state=one_hot_state, old_state=previous_state,
                                       threshold=loop_threshold)
            communicator.send_transition(cppu_name, obs, raw_action, reward, next_obs, done)
        if custom_reward == 'reward5':
            trajectory_for_semi_MDP.append({'step': step, 'old_state': old_state,
                                            'action': action, 'reward': reward,
                                            'new_state': state,
                                            'action_selected_by_algorithm':
                                                action_selected_by_algorithm})

        if action_selected_by_algorithm:
            agent_num = old_state['current_agent']
            agents_rewards[agent_num].append(reward)
            # check: verifica se è corretto old_state
            if custom_reward != 'reward5':
                agents_rewards_for_plot[agent_num] +=\
                    np.power(discount_factor, old_state['time']) * reward

            with open(file_log_name, 'a') as file:
                source_agent = old_state['current_agent']
                product = np.argmax(old_state['agents_state'][
                                        old_state['current_agent']])
                for i, sublist in enumerate(state['agents_state']):
                    if sublist[product] == 1:
                        dest_agent = i
                        break
                file.write(f"Agent {source_agent} transfer product {product} to"
                           f" agent {dest_agent} (action: {action})\n")
                
        if output_log and custom_reward != 'reward5':
            with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
                file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

        if export_trajectories:
            # update trajectory
            if action != nothing_action:
                with open(f"{TRAJECTORY_PATH}_{episode}.json", 'r') as infile:
                    trajectories = json.load(infile)
                state_to_save = copy.deepcopy(old_state)
                del state_to_save['time'], state_to_save['current_agent'],\
                    state_to_save['agents_busy']
                state_to_save['action_mask'] = {key: value.tolist()
                if isinstance(value, np.ndarray) else value for key, value in
                                            state_to_save['action_mask'].items()}
                state_to_save = {key: value.tolist() if
                isinstance(value, np.ndarray) else value for key, value in
                                 state_to_save.items()}
                trajectory_update = {'time': int(old_state['time']),
                                     'agent': int(old_state['current_agent']),
                                     'state': state_to_save, 'action': int(action),
                                     'reward': int(reward)}
                current_product = np.argmax(
                    state['agents_state'][old_state['current_agent']]) if \
                    action == 0 else np.argmax(
                    old_state['agents_state'][old_state['current_agent']])
                trajectories[f"Episode {episode}," \
                             f" Product {current_product}"].append(
                                                            trajectory_update)
                
                with open(f"{TRAJECTORY_PATH}_{episode}.json", 'w') as outfile:
                    json.dump(trajectories, outfile, indent=6)
        
        old_state = copy.deepcopy(state)
        if done:
            print(f"The episode {episode+1}/{n_episodes} is finished.")
            break
    if checkpoint_frequency is None:
        save_checkpoint = False
    else:
        save_checkpoint = True if (episode + 1) % checkpoint_frequency == 0 or episode == n_episodes - 1 else False
    communicator.publish_episode_management('End', episode_id, produced_product=None, save_checkpoint=save_checkpoint)
    communicator.sync_episode()
    # semi MDP reward postponed computation (at the end of the episode)
    if custom_reward == 'reward5':
        for actual_step in range(len(trajectory_for_semi_MDP) - 1):
            if trajectory_for_semi_MDP[actual_step]['action_selected_by_algorithm']:
                current_agent = trajectory_for_semi_MDP[actual_step]['old_state']['current_agent']
                actual_time = trajectory_for_semi_MDP[actual_step]['old_state']['time']
                next_step = actual_step + 1
                while next_step < (len(trajectory_for_semi_MDP) - 1) and \
                        trajectory_for_semi_MDP[next_step]['old_state']['agents_state'][current_agent][0] != 1:
                    next_step += 1
                next_time = trajectory_for_semi_MDP[next_step]['old_state']['time']
                trajectory_for_semi_MDP[actual_step]['reward'] = next_time - actual_time
        # semiMDP agents reward for plot
        for actual_step in trajectory_for_semi_MDP:
            if actual_step['action_selected_by_algorithm']:
                agents_rewards_for_plot[actual_step['old_state']['current_agent']] += \
                    np.power(discount_factor, actual_step['old_state']['time']) * actual_step['reward']
                        
        # semiMDP output log
        if output_log:
            with open(f"{OUTPUT_PATH}_{episode}.txt", 'a') as file:
                for actual_step in trajectory_for_semi_MDP:
                    file.write(f"***************Episode{episode}***************\n")
                    file.write(f"***************Step{actual_step['step']}***************\n")
                    file.write(f"Time: {actual_step['old_state']['time']}, Current agent: "
                               f"{actual_step['old_state']['current_agent']}\n")
                    file.write(f"Agents busy: {actual_step['old_state']['agents_busy']}\n")
                    file.write(f"Agents state: \n{actual_step['old_state']['agents_state']}\n")
                    file.write(f"Products state: \n{actual_step['old_state']['products_state']}\n")
                    file.write(f"Action mask: \n{actual_step['old_state']['action_mask']}\n")
                    file.write(f"Step: {step}, Action: {actual_step['action']}, Reward: {actual_step['reward']}\n\n")

    # if algorithm != 'random':
    #     if not test_model and update_values == 'episode':
    #         for agent in learning_agents:
    #             agent.apply_values_update()
    #
    #     if not test_model and policy_improvement == 'episode':
    #         for agent in learning_agents:
    #             agent.soft_policy_improvement()

    with open(file_log_name, 'a') as file:
        file.write(f"Episode {episode+1} ended\n")

    performance[episode] = {}
    performance[episode]['episode_duration'] = state['time']
    performance[episode]['agents_reward_for_plot'] = agents_rewards_for_plot

    for i in range(n_agents):
        mean_reward = np.mean(agents_rewards_for_plot[i])
        reward_visualizer.update_plot(i, episode, mean_reward)

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

reward_visualizer.save_plot(plot_path)
reward_visualizer.show_plot()

# if not test_model:
#     for agent in learning_agents:
#         agent.save()

if test_model:
    performance_log_name = f"{model_path}/test_performance.txt"
    if algorithm != 'random':
        plotter = DistQAndLPIPlotter(model_path, baseline_path)
        plotter.plot_reward_graphs()
        plotter.plot_performance_graph()
else:
    performance_log_name = f"{model_path}/training_performance.txt"

with open(performance_log_name, 'w') as file:
    episodes_time_to_complete = [performance[i]['episode_duration']
                                 for i in range(n_episodes)]
    file.write(f"Min time to complete: {np.min(episodes_time_to_complete)}\n")
    file.write(f"Avg time to complete: {np.average(episodes_time_to_complete)}\n")
    file.write(f"Max time to complete: {np.max(episodes_time_to_complete)}\n")
    file.write(f"Variance of time to complete: {np.var(episodes_time_to_complete)}\n")
    for j in range(n_agents):
        file.write(f"Agent {j} mean reward: "
                   f"{np.nanmean([performance[i][j]['mean_reward'] for i in range(n_episodes)])}\n")
    file.write(f"\n")
    for i in range(n_episodes):
        file.write(f"****Episode {i+1}****\n")
        file.write(f"Mean reward:\n")
        for j in range(n_agents):
            file.write(f"Agent {j}: {performance[i][j]['mean_reward']}\n")
        file.write(f"Time to complete: {performance[i]['episode_duration']}\n\n")
        
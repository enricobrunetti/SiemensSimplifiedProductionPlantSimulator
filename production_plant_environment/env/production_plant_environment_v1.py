from typing import Any
import gymnasium
from gymnasium import spaces
from utils.production_plant_environment_log_utils import OutputGenerator
import copy
import numpy as np
import random

class ProductionPlantEnvironment():
    def __init__(self, config, run, model_path, semiMDP_reward_config = None):
        self.config = config
        self.semiMDP_reward_config = semiMDP_reward_config
        self.custom_reward = config['custom_reward']
        self.algorithm = self.config['algorithm']
        self.num_max_steps = self.config['num_max_steps']
        self.n_agents = self.config['n_agents']
        self.n_products = self.config['n_products']
        self.n_production_skills = self.config['n_production_skills']
        self.actions = self.config['actions']
        self.use_action_masks = self.config['use_action_masks']
        self.n_production_skills = config['n_production_skills']
        self.nothing_action = self.n_production_skills + 5
        self.defer_action = self.nothing_action - 1
        self.illegal_action_penalty = self.config['illegal_action_penalty']
        self.action_time = self.config['action_time']
        self.agents_skills_custom_duration = {
            int(outer_key): {
                int(inner_key): value 
                for inner_key, value in inner_dict.items()
            } 
            for outer_key, inner_dict in self.config['agents_skills_custom_duration'].items()
        }
        self.action_energy = self.config['action_energy']
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        self.action_mask = {}
        self.output_generator = OutputGenerator(config, run, model_path)
        self.output_generator.generate_log_file()

        # initially mask all actions for all agents
        for i in range(self.n_agents):
            self.action_mask[i] = np.zeros(len(self.actions))

        self.agents_connections = {int(k): v for k, v in self.config['agents_connections'].items()}
        self.agents_skills = {int(k): v for k, v in self.config['agents_skills'].items()}
        self.supply_agent = self.config['supply_agent']

        self.performance = {}
        self.episode = -1

    def reset(self):
        self.episode += 1
        self.performance[self.episode] = {}
        self.output_generator.start_new_episode_log(self.episode)

        self.current_agent = 0
        self.time = 0
        self.actual_algorithm_step = 0
        self.trajectory = []

        # for every agent we have a tuple in which the first element is 0 if the 
        # agent is free and 1 if the agent is busy. The second element report on which
        # time the agent will become free
        self.agents_busy = {i: (0,0) for i in range(self.n_agents)}

        # list of all products for which the production has not started yet
        self.waiting_products = np.array(range(self.n_products))
        self.n_completed_products = 0
        self.current_step = 0

        for agent in range(self.n_agents):
            self.action_mask[agent] = np.array([0 for _ in range(len(self.actions))])
        self.action_mask[self.supply_agent][0] = 1

        # agents state
        self.agents_state = np.array(self.config['agents_starting_state'])

        # produts state
        self.products_state = np.array(self.config['products_starting_state'])

        state = self._next_observation()
        self.old_state = copy.deepcopy(state)
        state, _, _, _ = self._perform_internal_steps(state)
        if not self.use_action_masks:
            state = copy.deepcopy(state)
            state['action_mask'] = self._hide_action_mask(state['action_mask'], self.current_agent)
        return state
    
    def _next_observation(self):
        return {'current_agent': self.current_agent, 'time': self.time, 'agents_state': self.agents_state, 'products_state': self.products_state, 'action_mask': self.action_mask, 'agents_busy': self.agents_busy}

    def _take_action(self, action):
        action_time = self._get_action_time(self.current_agent, action)
        # perform production skills
        # # agent with supply skill take a new product and perform supply action
        if action == 0:
            # if there are no more porducts to start the production on
            # then doesn't allow anymore that action
            waiting_products = list(self.waiting_products)
            if not waiting_products:
                self.action_mask[self.supply_agent] = np.zeros(len(self.actions))
            else :
                next_product = random.choice(waiting_products)
                waiting_products.remove(next_product)
                self.waiting_products = np.array(waiting_products)
                self.agents_state[self.supply_agent][next_product] = 1
                self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
                self.action_mask[self.supply_agent] = self.compute_mask(self.supply_agent, next_product)
                self.agents_busy[self.supply_agent] = (1, self.time + action_time)
            self.output_generator.pick_up_new_product_log(self.current_agent, next_product)
        elif action < self.n_production_skills:
            actual_product = np.argmax(self.agents_state[self.current_agent] == 1)
            self.products_state[actual_product] = np.maximum(0, self.products_state[actual_product] - 1)
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, actual_product)
            self.agents_busy[self.current_agent] = (1, self.time + action_time)
            self.output_generator.production_skill_log(action, actual_product, self.current_agent)
        # perform transfer actions
        elif action < self.n_production_skills+4:
            next_agent = self._get_next_agent(self.current_agent, action)
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + action_time)
            self.output_generator.transfer_action_log(self.current_agent, np.argmax(self.agents_state[next_agent] == 1), next_agent, action)
        elif action == self.n_production_skills+4:
            self.agents_busy[self.current_agent] = (1, self.time + action_time)
            self.output_generator.transfer_action_log(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1), self.current_agent, action)
        # all the actions have been masked -> no action available (agent standby)
        else:
            pass

        # if we have transfered from supply agent then allow it to get a new product
        if action >= self.n_production_skills and action < self.n_production_skills+4 and self.current_agent == self.supply_agent:
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent)

        self.update_trasnfer_mask()
        
        # increase time for every action in which we don't have an empty agent doing nothing
        if (not (action == self.actions[-1] and self.agents_busy[self.current_agent][0] == 0)):
            self.time += 1

        # return as reward the execution time of the action
        return action_time * self.alpha + self.action_energy[action] * self.beta

    # called for all actions: decided by the algortihm, production skill and nothing actions
    def _internal_step(self, action, action_selected_by_algorithm = False):
        # check if action is allowed by action mask
        if self.action_mask[self.current_agent][action] == 1 or action == self.nothing_action:
            reward = self._take_action(action)

            # check if some agents now are free
            for agent in self.agents_busy:
                if self.agents_busy[agent][0] == 1 and self.agents_busy[agent][1] <= self.time:
                    self.agents_busy[agent] = (0, 0)
                    # if the production of a product is terminated remove it from agents_state and update the action mask
                    # of the agent that termined the product
                    if np.max(self.agents_state[agent]) == 1 and all(elem == 0 for elem in np.array(self.products_state[np.argmax(self.agents_state[agent])]).flatten()):
                        # TO-DO: find a cleaner way to produce this log
                        print(f"Product {np.argmax(self.agents_state[agent])} finished.")
                        self.agents_state[agent] = np.zeros_like(self.agents_state[agent])
                        self.action_mask[agent] = self.compute_mask(agent)

            if np.all(self.products_state == 0) and all(self.agents_busy[elem][1] == 0 for elem in self.agents_busy):
                done = True
                if self.semiMDP_reward_config and self.custom_reward == 'reward5':
                    self.trajectory = self._compute_semi_MDP_reward(self.trajectory)
                self.output_generator.end_episode_log(self.episode)
                self.output_generator.generate_outputs(self.trajectory, self.episode)
                self._compute_performance_end_of_episode()
            else:
                done = False

            if self.current_agent < self.n_agents - 1:
                self.current_agent += 1
            else :
                self.current_agent = 0

            if done:
                self.render_episode()
            
            info = {'illegal_action': False}
        # the action is illegal
        else:
            info = {'illegal_action': True}
            reward = -self.illegal_action_penalty
            done = False

        obs = self._next_observation()
        self.new_state = copy.deepcopy(obs)
        self.trajectory.append(({'step': self.current_step, 'old_state': self.old_state, 'action': action, 'illegal_action': info["illegal_action"], 'reward': reward, 'new_state': self.new_state, 'action_selected_by_algorithm': action_selected_by_algorithm}))
        self.old_state = copy.deepcopy(self.new_state)
        self.current_step += 1
        return obs, reward, done, info
    
    # called only for actions decided by the algorithm
    def step(self, action):
        agent = self.current_agent
        product = np.argmax(self.agents_state[agent])
        products_state = copy.deepcopy(self.products_state)
        
        state, reward, done, info = self._internal_step(action, True)
        
        if not info['illegal_action']:
            state, reward, done, _ = self._perform_internal_steps(state, done)
            info = self._check_production_skill_perfomed_by_next_agent(agent, action, product, products_state)
        if not self.use_action_masks:
            state = copy.deepcopy(state)
            state['action_mask'] = self._hide_action_mask(state['action_mask'], self.current_agent)

        self.actual_algorithm_step += 1
        if self.actual_algorithm_step >= self.num_max_steps:
            truncation = True
            state = copy.deepcopy(state)
            state['time'] = self.num_max_steps
            if self.semiMDP_reward_config and self.custom_reward == 'reward5':
                self.trajectory = self._compute_semi_MDP_reward(self.trajectory)
                self.output_generator.end_episode_log(self.episode)
                self.output_generator.generate_outputs(self.trajectory, self.episode)
                self._compute_performance_end_of_episode(truncation = True)
        else:
            truncation = False

        return state, reward, done, truncation, info

    def _perform_internal_steps(self, state, done = 0):
        start_time = state['time'] - 1
        action_selected_by_algorithm_needed = False
        while not action_selected_by_algorithm_needed:
            if np.all(np.array(state['action_mask'][state['current_agent']]) == 0) or state['agents_busy'][state['current_agent']][0] == 1:
                # if no actions available -> do nothing
                action = self.nothing_action
            else:
                actions = np.array(self.actions)
                actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
                # if only production actions are available there will be only one element
                if np.max(actions) < self.n_production_skills:
                    action = actions[0]
                else :
                    action_selected_by_algorithm_needed = True
            if action_selected_by_algorithm_needed:
                end_time = state['time']
                reward = end_time - start_time
                return state, reward, done, {}
            else:
                state, reward, done, _ = self._internal_step(action)
                if done:
                    end_time = state['time']
                    reward = end_time - start_time
                    return state, reward, done, {}

    def _check_production_skill_perfomed_by_next_agent(self, agent, action, product, products_state):
        if action == self.defer_action:
            return {'production_skill_executed': False, 'transport_duration': self._get_action_time(agent, action)}
        else:
            next_agent = self._get_next_agent(agent, action)
            next_skill = [i for i in range(len(products_state[product])) if 1 in products_state[product][i]][0]
            if next_skill in self.agents_skills[next_agent]:
                return {'production_skill_executed': True, 'transport_duration': self._get_action_time(agent, action), 'production_skill_duration': self._get_action_time(next_agent, next_skill)}
            else:
                return {'production_skill_executed': False, 'transport_duration': self._get_action_time(agent, action)}

    def _get_next_agent(self, agent, action):
        return self.agents_connections[agent][action - self.n_production_skills]
    
    def _get_action_time(self, agent, action):
        # compute time needed to perform the action (if there is custom one for that agent use it otherwise default one for that specific action)
        if agent in self.agents_skills_custom_duration and action in self.agents_skills_custom_duration[agent]:
            return self.agents_skills_custom_duration[agent][action]
        else:
            return self.action_time[action]

    # TO-DO: see if it makes sens to add also another print on the console
    def render_episode(self):
        pass

    def compute_mask(self, agent, product = None):
        # if the agent has a product do that
        if product != None:
            mask = self.compute_production_skill_masking(agent, product)
            if np.all(mask == 0):
                mask = np.append(mask, self.compute_transfer_masking(agent))
                # allow defer action
                mask = np.append(mask, 1)
            else:
                mask = np.append(mask, np.zeros(5))
            # don't allow do nothing
            mask = np.append(mask, 0)
        # if the agent has no product mask all actions as zeroes unless it is the supply agent 
        # (in that case mask the take a new element action as 1)
        else:
            mask = np.zeros(len(self.actions))
            if agent == self.supply_agent and self.waiting_products.size > 0:
                mask[0] = 1
        return mask

    # return an array containing n_production_skills elements, one corresponding to the next skill
    # to perform on the product (if availvale for that agent) and zeroes corresponding 
    # to the skills that aren't the next one or can't be performed
    def compute_production_skill_masking(self, agent, product):
        mask = np.zeros(self.n_production_skills)
        skills = [i for i in range(len(self.products_state[product])) if 1 in self.products_state[product][i]]
        skill = skills[0] if len(skills) > 0 else 0
        if skill in self.agents_skills[agent]:
            mask[skill] = 1
        return mask

    # return an array containing 4 elements, ones corrensponding to
    # directions in which there is another agent connected and zeroes
    # corresponding to directions where there is no other agent
    def compute_transfer_masking(self, agent):
        mask = []
        for elem in self.agents_connections[agent]:
            if elem != None and np.all(self.agents_state[elem] == 0):
                elem = 1
            else:
                elem = 0
            mask = np.append(mask, elem)
        return mask
    
    # TO-DO: do that only when there is a transfer in order to speed up (you need to do that for everyone is connected
    # to the destination of the transport except for the destination of the transport)
    # update the action mask for transfer actions in order to avoid sending a product to
    # an agent that already has another product
    def update_trasnfer_mask(self):
        for agent in self.agents_connections:
            if self.hasProduct(agent):
                for i in range(self.n_production_skills, self.n_production_skills+4):
                    if self.agents_connections[agent][i - self.n_production_skills] != None and self.hasProduct(self.agents_connections[agent][i - self.n_production_skills]):
                        self.action_mask[agent][i] = 0
                    elif self.agents_connections[agent][i - self.n_production_skills] != None and (not self.hasProduct(self.agents_connections[agent][i - self.n_production_skills]) and np.all(self.action_mask[agent][:self.n_production_skills] == 0)):
                        self.action_mask[agent][i] = 1

    # If called remove masks due to an occupied near agent. It keeps all masks related to the structure of the scenario
    def _hide_action_mask(self, action_mask, agent):
        for i in range(self.n_production_skills, self.n_production_skills + 4):
            if self.agents_connections[agent][i - self.n_production_skills] != None:
                action_mask[agent][i] = 1
        return action_mask

    # utility function for update_transfer_mask
    def hasProduct(self, agent):
        return not np.all(self.agents_state[agent] == 0)
    
    def _compute_performance_end_of_episode(self, truncation = False):
        agents_rewards = self.get_agents_rewards_for_semiMDP()
        if truncation:
            self.performance[self.episode]['episode_duration'] = self.num_max_steps
        else:
            self.performance[self.episode]['episode_duration'] = self.time
        self.performance[self.episode]['agents_reward_for_plot'] = agents_rewards

        for i in range(self.n_agents):
            mean_reward = np.mean(agents_rewards[i])

            self.performance[self.episode][i] = {}
            self.performance[self.episode][i]['mean_reward'] = mean_reward
        
        self.output_generator.generate_performance_log(self.performance, self.episode)
    
    def _compute_semi_MDP_reward2(self, trajectory_for_semi_MDP):
        for actual_step in range(len(trajectory_for_semi_MDP) - 1):
            if trajectory_for_semi_MDP[actual_step]['action_selected_by_algorithm']:
                current_agent = trajectory_for_semi_MDP[actual_step]['old_state']['current_agent']
                current_product = np.where(trajectory_for_semi_MDP[actual_step]['old_state']['agents_state'][current_agent] == 1)[0][0]
                actual_time = trajectory_for_semi_MDP[actual_step]['old_state']['time']
                actual_action = trajectory_for_semi_MDP[actual_step]['action']
                next_step = actual_step + 1
                while next_step < (len(trajectory_for_semi_MDP) - 1) and trajectory_for_semi_MDP[next_step]['old_state']['agents_state'][current_agent][current_product] != 1:
                    next_step += 1
                next_time = trajectory_for_semi_MDP[next_step]['old_state']['time']
                trajectory_for_semi_MDP[actual_step]['reward'] = -1 * (next_time - actual_time)
                if actual_action == self.defer_action:
                    next_agent = current_agent
                else:
                    next_agent = self._get_next_agent(current_agent, actual_action)
                if self.semiMDP_reward_config['positive_shaping']:
                    for check_positive_shaping_step in range(len(trajectory_for_semi_MDP) - actual_step - 1):
                        if trajectory_for_semi_MDP[actual_step + check_positive_shaping_step + 1]['old_state']['current_agent'] == next_agent:
                            next_agent_action = trajectory_for_semi_MDP[actual_step + check_positive_shaping_step + 1]['action']
                            if next_agent_action != self.nothing_action:
                                if next_agent_action < self.n_production_skills:
                                    next_agent_action_time = self._get_action_time(next_agent, next_agent_action)
                                    trajectory_for_semi_MDP[actual_step]['reward'] = next_agent_action_time * self.semiMDP_reward_config['positive_shaping_constant']
                                break
                if actual_action == self.defer_action and self.semiMDP_reward_config['negative_shaping']:
                    trajectory_for_semi_MDP[actual_step]['reward'] = -1 * self.semiMDP_reward_config['negative_shaping_constant']
        
        return trajectory_for_semi_MDP
    
    def _compute_semi_MDP_reward(self, trajectory_for_semi_MDP):
        terminated_products = []
        semiMDP_computer = {}
        for actual_step in range(len(trajectory_for_semi_MDP)):
            current_agent = trajectory_for_semi_MDP[actual_step]['old_state']['current_agent']
            current_product_list = np.where(trajectory_for_semi_MDP[actual_step]['old_state']['agents_state'][current_agent] == 1)[0]
            if len(current_product_list) > 0:
                has_product = True
                current_product = current_product_list[0]
            else :
                has_product = False
            actual_action = trajectory_for_semi_MDP[actual_step]['action']
            if has_product and trajectory_for_semi_MDP[actual_step]['old_state']['agents_busy'][current_agent][0] == 0:
                actual_time = trajectory_for_semi_MDP[actual_step]['old_state']['time']
                if current_agent in semiMDP_computer and current_product in semiMDP_computer[current_agent]:
                    last_seen_step = semiMDP_computer[current_agent][current_product]['last_seen_step']
                    last_seen_time = semiMDP_computer[current_agent][current_product]['last_seen_time']
                    new_reward = -1 * (actual_time - last_seen_time)
                    new_reward += semiMDP_computer[current_agent][current_product]['shaping']
                    trajectory_for_semi_MDP[last_seen_step]['reward'] = new_reward
                    semiMDP_computer[current_agent][current_product]['last_seen_step'] = actual_step
                    semiMDP_computer[current_agent][current_product]['last_seen_time'] = actual_time
                    semiMDP_computer[current_agent][current_product]['shaping'] = 0 
                else:
                    if current_agent not in semiMDP_computer:
                        semiMDP_computer[current_agent] = {}
                    semiMDP_computer[current_agent][current_product] = {}
                    semiMDP_computer[current_agent][current_product]['last_seen_step'] = actual_step
                    semiMDP_computer[current_agent][current_product]['last_seen_time'] = actual_time
                    semiMDP_computer[current_agent][current_product]['shaping'] = 0 

            if self.semiMDP_reward_config['positive_shaping'] and actual_action < self.n_production_skills and has_product:
                action_time = self._get_action_time(current_agent, actual_action)
                for agent in range(self.n_agents):
                    if agent in semiMDP_computer and current_product in semiMDP_computer[agent]:
                        if self.semiMDP_reward_config['positive_shaping_equal']:
                            semiMDP_computer[agent][current_product]['shaping'] += self.semiMDP_reward_config['positive_shaping_constant']
                        else:
                            semiMDP_computer[agent][current_product]['shaping'] += (action_time * self.semiMDP_reward_config['positive_shaping_constant'])

            if self.semiMDP_reward_config['negative_shaping'] and actual_action == self.defer_action and has_product:
                semiMDP_computer[current_agent][current_product]['shaping'] -= 1 * self.semiMDP_reward_config['negative_shaping_constant']

            if trajectory_for_semi_MDP[actual_step]['illegal_action']:
                semiMDP_computer[current_agent][current_product]['shaping'] -= 1 * self.semiMDP_reward_config['illegal_action_penalty']
            
            if not self.semiMDP_reward_config['semiMDP_till_end_of_episode']:
                agents_busy = trajectory_for_semi_MDP[actual_step]['old_state']['agents_busy']
                products_state = trajectory_for_semi_MDP[actual_step]['old_state']['products_state']
                agents_state = trajectory_for_semi_MDP[actual_step]['old_state']['agents_state']
                for agent in agents_busy:
                    if agents_busy[agent][0] == 1 and agents_busy[agent][1] == trajectory_for_semi_MDP[actual_step]['old_state']['time'] + 1:
                        prod = np.argmax(agents_state[agent])
                        if np.max(agents_state[agent]) == 1 and all(elem == 0 for elem in np.array(products_state[prod]).flatten()) and prod not in terminated_products:
                            final_time = trajectory_for_semi_MDP[actual_step]['old_state']['time']
                            terminated_products.append(prod)
                            for a in range(self.n_agents):
                                if a in semiMDP_computer and prod in semiMDP_computer[a]:
                                    last_seen_step = semiMDP_computer[a][prod]['last_seen_step']
                                    last_seen_time = semiMDP_computer[a][prod]['last_seen_time']
                                    new_reward = -1 * (final_time - last_seen_time)
                                    new_reward += semiMDP_computer[a][prod]['shaping']
                                    trajectory_for_semi_MDP[last_seen_step]['reward'] = new_reward

        if self.semiMDP_reward_config['semiMDP_till_end_of_episode']:
            final_time = trajectory_for_semi_MDP[-1]['old_state']['time']
            for agent in range(self.n_agents):
                for product in range(self.n_products):
                    if agent in semiMDP_computer and product in semiMDP_computer[agent]:
                        last_seen_step = semiMDP_computer[agent][product]['last_seen_step']
                        last_seen_time = semiMDP_computer[agent][product]['last_seen_time']
                        new_reward = -1 * (final_time - last_seen_time)
                        new_reward += semiMDP_computer[agent][product]['shaping']
                        trajectory_for_semi_MDP[last_seen_step]['reward'] = new_reward
        
        return trajectory_for_semi_MDP

    # call it at the end of an episode before reset method
    def get_actual_trajectory(self):
        return self.trajectory
    
    # call it only is semiMDP reward is selected
    def get_agents_rewards_for_semiMDP(self):
        trajectory_for_semi_MDP = copy.deepcopy(self.trajectory)
        agents_rewards = {}
        for i in range(self.n_agents):
            agents_rewards[i] = 0
        for actual_step in trajectory_for_semi_MDP:
            if actual_step['action_selected_by_algorithm']:
                agents_rewards[actual_step['old_state']['current_agent']] += np.power(self.config['gamma'],  actual_step['old_state']['time']) * actual_step['reward']
        return agents_rewards


    



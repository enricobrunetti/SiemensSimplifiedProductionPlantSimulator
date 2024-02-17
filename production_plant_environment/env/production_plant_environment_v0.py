from typing import Any
import gymnasium
from gymnasium import spaces

import numpy as np
import random

class ProductionPlantEnvironment():
    def __init__(self, config = None):
        self.config = config
        self.n_agents = self.config['n_agents']
        self.n_products = self.config['n_products']
        self.n_production_skills = self.config['n_production_skills']
        self.action_space = self.config['actions']
        self.action_time = self.config['action_time']
        self.action_energy = self.config['action_energy']
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        self.action_mask = {}

        # initially mask all actions for all agents
        for i in range(self.n_agents):
            self.action_mask[i] = np.zeros(len(self.action_space))

        self.agents_connections = {int(k): v for k, v in self.config['agents_connections'].items()}
        self.agents_skills = {int(k): v for k, v in self.config['agents_skills'].items()}
        self.supply_agent = self.config['supply_agent']

    def reset(self):
        self.current_agent = 0
        self.time = 0

        # for every agent we have a tuple in which the first element is 0 if the 
        # agent is free and 1 if the agent is busy. The second element report on which
        # time the agent will become free
        self.agents_busy = {i: (0,0) for i in range(self.n_agents)}

        # list of all products for which the production has not started yet
        self.waiting_products = np.array(range(self.n_products))
        self.n_completed_products = 0
        self.current_step = 0

        self.action_mask[self.current_agent] = [0 for i in range(len(self.action_space))]
        self.action_mask[self.current_agent][0] = 1

        # agents state
        self.agents_state = np.array(self.config['agents_starting_state'])

        # produts state
        self.products_state = np.array(self.config['products_starting_state'])
        
        return self._next_observation()
    
    def _next_observation(self):
        return {'current_agent': self.current_agent, 'time': self.time, 'agents_state': self.agents_state, 'products_state': self.products_state, 'action_mask': self.action_mask, 'agents_busy': self.agents_busy}

    def _take_action(self, action):
        # perform production skills
        # # agent with supply skill take a new product and perform supply action
        if action == 0:
            # if there are no more porducts to start the production on
            # then doesn't allow anymore that action
            waiting_products = list(self.waiting_products)
            if not waiting_products:
                self.action_mask[self.supply_agent] = np.zeros(len(self.action_space))
            else :
                next_product = random.choice(waiting_products)
                waiting_products.remove(next_product)
                self.waiting_products = np.array(waiting_products)
                self.agents_state[self.supply_agent][next_product] = 1
                self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
                self.action_mask[self.supply_agent] = self.compute_mask(self.supply_agent, next_product)
                self.agents_busy[self.supply_agent] = (1, self.time + self.action_time[action])
        elif action < self.n_production_skills:
            self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1))
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        # perform transfer actions
        elif action < self.n_production_skills+4:
            next_agent = self.agents_connections[self.current_agent][action - self.n_production_skills]
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + self.action_time[action])
        elif action == self.n_production_skills+4:
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        # all the actions have been masked -> no action available (agent standby)
        else:
            pass

        # if we have transfered from supply agent then allow it to get a new product
        if action >= self.n_production_skills and action < self.n_production_skills+4 and self.current_agent == self.supply_agent:
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent)

        self.update_trasnfer_mask()
        
        # increase time for every action in which we don't have an empty agent doing nothing
        if (not (action == self.action_space[-1] and self.agents_busy[self.current_agent][0] == 0)):
            self.time += 1

        # return as reward the execution time of the action
        return self.action_time[action] * self.alpha + self.action_energy[action] * self.beta

    def step(self, action):
        reward = self._take_action(action)
        self.current_step += 1

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
        else:
            done = False

        if self.current_agent < self.n_agents - 1:
            self.current_agent += 1
        else :
            self.current_agent = 0

        if done:
            self.render_episode()

        obs = self._next_observation()

        return obs, reward, done, {}
    
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
            mask = np.zeros(len(self.action_space))
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

    # utility function for update_transfer_mask
    def hasProduct(self, agent):
        return not np.all(self.agents_state[agent] == 0)


    



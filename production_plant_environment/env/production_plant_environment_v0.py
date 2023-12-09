from typing import Any
import gymnasium
from gymnasium import spaces

import numpy as np
import random

class ProductionPlantEnvironment():
    def __init__(self):
        self.n_agents = 5
        self.n_products = 4
        self.n_production_skills = 4
        self.action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.action_time =  [5, 5, 5, 5, 2, 2, 2, 2, 0, 0, 2]
        self.action_mask = {}

        # initially mask all actions for all agents
        for i in range(self.n_agents):
            self.action_mask[i] = np.zeros(len(self.action_space))

        self.agents_connections = {0: [None, 1, 2, None],
                                   1: [None, 4, 2, 0],
                                   2: [1, 3, None, 0],
                                   3: [4, None, None, 2],
                                   4: [None, None, 3, 1]}
        
        self.agents_skills = {0: [0], 1: [1, 3], 2: [2, 3], 3: [1, 3], 4: [2, 3]}

        self.supply_agent = 0

        self.reward_range = (-200, 200)

        self.current_episode = 0
        self.success_episode = []

    def reset(self):
        self.current_agent = 0
        self.time = 0

        # for every agent we have a tuple in which the first element is 0 if the 
        # agent is free and 1 if the agent is busy. The second element report on which
        # time the agent will become free
        self.agents_busy = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0)}

        # list of all products for which the production has not started yet
        self.waiting_products = np.array(range(self.n_products))
        self.n_completed_products = 0
        self.current_step = 0
        self.max_step = 100

        self.action_mask[self.current_agent] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        #agents state
        self.agents_state = np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])
        
        self.products_state = np.array([[1, 2, 3, 4],
                                        [1, 0, 2, 3],
                                        [1, 3, 2, 4],
                                        [1, 0, 0, 2]])
        
        return self._next_observation()
    
    def _next_observation(self):
        obs = {'current_agent': self.current_agent, 'time': self.time, 'agents_state': self.agents_state, 'products_state': self.products_state, 'action_mask': self.action_mask, 'agents_busy': self.agents_busy}
        return obs

    def _take_action(self, action):
        # TO-DO: try to accorpate production skills and transfer actions
        # perform production skills
        if action == 0:
            self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
            action_result = 10
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1))
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        elif action == 1:
            self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
            action_result = 10
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1))
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        elif action == 2:
            self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
            action_result = 10
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1))
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        elif action == 3:
            self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] = np.maximum(0, self.products_state[np.argmax(self.agents_state[self.current_agent] == 1)] - 1)
            action_result = 10
            self.action_mask[self.current_agent] = self.compute_mask(self.current_agent, np.argmax(self.agents_state[self.current_agent] == 1))
            self.agents_busy[self.current_agent] = (1, self.time + self.action_time[action])
        # perform transfer actions
        elif action == 4:
            next_agent = self.agents_connections[self.current_agent][0]
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            action_result = 5
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + self.action_time[action])
        elif action == 5:
            next_agent = self.agents_connections[self.current_agent][1]
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            action_result = 5
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + self.action_time[action])
        elif action == 6:
            next_agent = self.agents_connections[self.current_agent][2]
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            action_result = 5
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + self.action_time[action])
        elif action == 7:
            next_agent = self.agents_connections[self.current_agent][3]
            self.agents_state[next_agent] = self.agents_state[self.current_agent]
            self.agents_state[self.current_agent] = np.zeros_like(self.agents_state[self.current_agent])
            action_result = 5
            self.action_mask[self.current_agent] = np.zeros_like(self.action_mask[self.current_agent])
            self.action_mask[next_agent] = self.compute_mask(next_agent, np.argmax(self.agents_state[next_agent] == 1))
            self.agents_busy[next_agent] = (1, self.time + self.action_time[action])
        elif action == 8:
            action_result = 0
        # agent with supply skill take a new product
        elif action == 10:
            # if there are no more porducts to start the production on
            # then doesn't allow anymore that action
            waiting_products = list(self.waiting_products)
            if not waiting_products:
                self.action_mask[self.supply_agent] = np.zeros(len(self.action_space))
                action_result = 0
            else :
                next_product = random.choice(waiting_products)
                waiting_products.remove(next_product)
                self.waiting_products = np.array(waiting_products)
                self.agents_state[self.supply_agent][next_product] = 1
                action_result = 10
                self.action_mask[self.supply_agent] = self.compute_mask(self.supply_agent, next_product)
                self.agents_busy[self.supply_agent] = (1, self.time + self.action_time[action])
        # all the actions have been masked -> no action available (agent standby)
        else:
            action_result = 0

        # if we have transfered from supply agent then allow it to get a new product
        if action >= 4 and action <= 7 and self.current_agent == self.supply_agent:
            self.action_mask[self.current_agent][-1] = 1

        self.update_trasnfer_mask()
        
        # CHECK IF IT WORKS
        # increase time for every action in which we don't have an empty agent doing nothing
        if (not (action == 9 and self.agents_busy[self.current_agent][0] == 0)):
            self.time += 1
        return action_result

    def step(self, action):
        reward = self._take_action(action)
        self.current_step += 1

        # check if some agents now are free
        for agent in self.agents_busy:
            if self.agents_busy[agent][0] == 1 and self.agents_busy[agent][1] <= self.time:
                self.agents_busy[agent] = (0, 0)
                # if the production of a product is terminated remove it from agents_state and update the action mask
                # of the agent that termined the product
                if np.max(self.agents_state[agent]) == 1 and all(elem == 0 for elem in self.products_state[np.argmax(self.agents_state[agent])]):
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
    
    # TO-DO: see if it makes senso to add also another print on the console
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
            mask = np.append(mask, [0, 0])
        # if the agent has no product mask all actions as zeroes unless it is the supply agent 
        # (in that case mask the take a new element action as 1)
        else:
            mask = np.zeros(len(self.action_space))
            if agent == self.supply_agent:
                mask[-1] = 1
        return mask

    # return an array containing 4 elements, one corresponding to the next skill
    # to perform on the product (if availvale for that agent) and zeroes corresponding 
    # to the skills that aren't the next one or can't be performed
    def compute_production_skill_masking(self, agent, product):
        mask = np.zeros(self.n_production_skills)
        skill = np.argmax(self.products_state[product] == 1)
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
    
    # update the action mask for transfer actions in order to avoid sending a product to
    # an agent that already has another product
    def update_trasnfer_mask(self):
        for agent in self.agents_connections:
            for i in range(4, 7):
                if self.agents_connections[agent][i - 4] != None and not all(elem == 0 for elem in self.agents_state[self.agents_connections[agent][i - 4]]) and agent in self.action_mask:
                    self.action_mask[agent][i] = 0


    



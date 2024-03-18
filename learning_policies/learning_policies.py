import numpy as np
import json
import os

class LearningAgent:
    def __init__(self, config, agent_num, n_training_episodes, reward_type):
        self.algorithm = config['algorithm']
        self.agent_num = agent_num
        self.n_training_episodes = n_training_episodes
        self.reward_type = reward_type
        self.actions = config['available_actions']
        self.actions_space = len(config['available_actions'])
        self.n_products = config['n_products']
        self.agents_connections = {int(k): v for k, v in config['agents_connections'].items()}

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.eta = config['eta']
        self.tau = config['tau']
        
        self.default_q_value = np.round(-self.n_products/(1 - self.gamma), 3)

        self.values = {}
        self.policy = {}

        # used to allow to keep track of updates and transfer to actual values
        # only once an episode is finished 
        self.values_updated = {}

        self.p_max = config['p_max']

        self.actions_policy = config['actions_policy']
        self.exploration_prob = config['exploration_prob']

        self.model_name = f'models/{self.reward_type}/{self.algorithm}/{self.algorithm}_{self.actions_policy}_{self.n_training_episodes}_{self.alpha}_{self.gamma}_{self.eta}_{self.tau}'

    def apply_values_update(self):
        self.values = self.values_updated

    def select_action(self, observation, mask):
        allowed_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        rescaled_non_production_actions = [action - self.actions[0] for action in allowed_actions]
        
        if self.actions_policy == 'eps-greedy':
            if np.random.rand() < self.exploration_prob:
                print('random action choosen')
                return self.get_random_action(allowed_actions)
            else:
                if observation not in self.values.keys():
                    print('new observation, random action selection')
                    return self.get_random_action(allowed_actions)
                
                q_values = [self.values[observation]['Q'][a] for a in rescaled_non_production_actions]
                print(f'actions: {allowed_actions}, q_values: {q_values}')
                return allowed_actions[np.argmax(q_values)]
            
        elif self.actions_policy == 'softmax':
            if observation not in self.policy.keys():
                print('new observation, random action selection')
                return self.get_random_action(allowed_actions)
            
            # IMPORTANT: we are computing probability referred only to allowed actions
            policy_values = [self.policy[observation][a] for a in rescaled_non_production_actions]
            prob_values = policy_values / np.sum(policy_values)
            print(f'actions: {allowed_actions}, prob_values: {prob_values}')
            return np.random.choice(allowed_actions, p=prob_values)
    
    def save(self):
        model_agent_path = f'{self.model_name}/{self.agent_num}.json'
        
        directory = os.path.dirname(model_agent_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        policy_to_save = {}
        for key, value in self.policy.items():
            policy_to_save[key] = list(value)
        values_and_policy = {'Values': self.values, 'Policy': policy_to_save}
        with open(model_agent_path, 'w') as outfile:
            json.dump(values_and_policy, outfile, indent=6)
        
    def load(self, model_name):
        model_dir = f'{model_name}.json'
        with open(model_dir, 'r') as infile:
            trajectories = json.load(infile)
        self.values = trajectories['Values']
        self.policy = trajectories['Policy']

    def get_model_name(self):
        return self.model_name
    
    def get_gamma(self):
        return self.gamma

class DistributedQLearningAgent(LearningAgent):
    def __init__(self, config, agent_num, n_training_episodes, reward_type):
        super().__init__(config, agent_num, n_training_episodes, reward_type)

    def update_values(self, observation, action, reward, agents_information):
        action = action - self.actions[0]
    
        if observation not in self.values_updated.keys():
            self.values_updated[observation] = {}
            self.values_updated[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values_updated[observation]['T'] = [0] * self.actions_space

            # in that case the observation must be added also in the policy
            if self.actions_policy == 'softmax':
                self.policy[observation] = 1/self.actions_space * np.ones(self.actions_space)

        lr = self.alpha

        current_q_value = self.values_updated[observation]['Q'][action]
        next_q_value = (1 - lr) * current_q_value + lr * (reward + self.gamma * agents_information)

        self.values_updated[observation]['Q'][action] = np.round(next_q_value, 3)
        self.values_updated[observation]['T'][action] += 1

    def soft_policy_improvement(self):
        # otherwise policy improvement not needed
        if self.actions_policy == 'softmax':
            lr_policy = self.eta
            for _ in range(self.p_max):
                for state in self.policy.keys():
                    curr_value = self.policy[state]
                    sum_curr_value = np.sum(curr_value)
                    new_value = (curr_value ** (1 - lr_policy * self.tau) / sum_curr_value) * np.exp(lr_policy * np.array(self.values[state]['Q']))
                    self.policy[state] = np.round(new_value, 3)
            
    def get_next_agent_number(self, action):
        action -= self.actions[0]
        if action == 4:
            return self.agent_num
        return self.agents_connections[self.agent_num][action]    

    def get_random_action(self, allowed_actions):
        return np.random.choice(allowed_actions)

    def get_max_value(self, observation):
        if observation not in self.values.keys():
            return self.default_q_value
        return np.max(self.values[observation]['Q'])

class LPIAgent(LearningAgent):
    def __init__(self, config, agent_num, n_training_episodes, reward_type):
        super().__init__(config, agent_num, n_training_episodes, reward_type)

        self.beta = config['beta']
        self.kappa = config['kappa']

        self.neighbours_kappa = self.get_n_hop_neighbours(self.kappa)
        self.neighbours_beta = self.get_n_hop_neighbours(self.beta)

        self.model_name = f'models/{self.reward_type}/{self.algorithm}/{self.algorithm}_{self.actions_policy}_{self.n_training_episodes}_{self.alpha}_{self.gamma}_{self.eta}_{self.tau}_{self.beta}_{self.kappa}'

    def get_n_hop_neighbours(self, n_hop):
        neighbour = set()
        for i in range(n_hop):
            if i == 0:
                neighbour = set(elem for elem in self.agents_connections[self.agent_num] if elem != None)
            else:
                for elem in neighbour:
                    new_neighbours = set(new_elem for new_elem in self.agents_connections[elem] if new_elem != None)
                    neighbour = neighbour.union(new_neighbours)
        
        if self.agent_num not in neighbour:
            neighbour.add(self.agent_num)

        return list(neighbour)

    # IMPORTANT: obs_prime and a_prime contains the observation of the same agent in his next turn and the action
    # that he will choose in that turn (so not next state in general but next state in which the agent has
    # again a product)
    def update_values(self, observation, action, reward, obs_prime, a_prime):
        lr = self.alpha

        action = action - self.actions[0]
        a_prime = a_prime - self.actions[0]

        if observation not in self.values_updated.keys():
            self.values_updated[observation] = {}
            self.values_updated[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values_updated[observation]['T'] = [0] * self.actions_space

            # in that case the observation must be added also in the policy
            if self.actions_policy == 'softmax':
                self.policy[observation] = 1/self.actions_space * np.ones(self.actions_space)

        curr_value = self.values_updated[observation]['Q'][action]
        # agents_information=q_value dell'azione (a_prime) che l'agente stesso (lui stesso non il successivo) giocherebbe 
        #nello stato s_prime (prossimo stato in cui l'agente ha un prodotto)
        agents_information = self.get_values(obs_prime)[a_prime]

        next_value = (1 - lr) * curr_value + lr * (reward + self.gamma * agents_information)

        self.values_updated[observation]['Q'][action] = np.round(next_value, 3)
        self.values_updated[observation]['T'][action] += 1

    def soft_policy_improvement(self):
        # otherwise policy improvement not needed
        if self.actions_policy == 'softmax':
            lr_policy = self.eta
            for _ in range(self.p_max):
                for state in self.policy.keys():
                    curr_value = self.policy[state]
                    sum_curr_value = np.sum(curr_value)
                    new_value = (curr_value ** (1 - lr_policy * self.tau) / sum_curr_value) * np.exp(lr_policy * np.array(self.values[state]['Q']))
                    self.policy[state] = np.round(new_value, 3)

    def generate_observation(self, state):

        combined_observation = []
        for i in range(len(state['agents_state'])):
            if i in self.neighbours_beta:
                single_obs = []
                single_obs.append(state['agents_state'][i])
                if all(elem == 0 for elem in single_obs[0]):
                    single_obs.append(np.zeros_like(state['products_state'][0]).tolist())
                else:
                    single_obs.append(state['products_state'][np.argmax(single_obs[0])])
                combined_observation.append(single_obs)

        return str(combined_observation)
        
    def get_random_action(self, allowed_actions):
        return np.random.choice(allowed_actions)
    
    def get_values(self, observation):
        if observation not in self.values.keys():
            return [self.default_q_value] * self.actions_space
        return self.values[observation]['Q']
    
    def save(self):
        model_agent_path = f'{self.model_name}/{self.agent_num}.json'
        
        directory = os.path.dirname(model_agent_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        policy_to_save = {}
        for key, value in self.policy.items():
            policy_to_save[key] = list(value)
        values_and_policy = {'Values': self.values, 'Policy': policy_to_save}
        with open(model_agent_path, 'w') as outfile:
            json.dump(values_and_policy, outfile, indent=6)

    def get_model_name(self):
        return self.model_name
    

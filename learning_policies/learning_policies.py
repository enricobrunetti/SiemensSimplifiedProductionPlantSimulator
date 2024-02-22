import numpy as np

# TO-DO: check if it is better to move this class somewhere else
class AgentsManager():
    def __init__(self, config):
        self.config = config
        self.agents = {}
        self.agents_connections = {int(k): v for k, v in config['agents_connections'].items()}
        self.actions = config['available_actions']
    
    def register_agent(self, agent):
        self.agents[agent.agent_num] = agent

    def get_values(self, agent_num, observation):
        values = self.agents[agent_num].values
        if observation not in values.keys():
            return None
        return values[observation]['Q']
    
    def get_max_value(self, agent_num, observation):
        values = self.agents[agent_num].values
        if observation not in values.keys():
            return None
        return np.max(values[observation]['Q'])
    
    def get_n_hop_neighbours(self, agent_num, n_hop):
        neighbour = set()
        for i in range(n_hop):
            if i == 0:
                neighbour = set(elem for elem in self.agents_connections[agent_num] if elem != None)
            else:
                for elem in neighbour:
                    new_neighbours = set(new_elem for new_elem in self.agents_connections[elem] if new_elem != None)
                    neighbour = neighbour.union(new_neighbours)
        
        if agent_num not in neighbour:
            neighbour.add(agent_num)

        return list(neighbour)
    
    # CHECK if necessary
    def get_default_observation(self, number_of_agents):
        null_observation = []
        null_observation.append([None]) # agent state
        null_observation.append([None]) # production skills for the product that the agent has

        null_observation = [tuple(null_observation) for _ in range (number_of_agents)]
        null_action = [self.actions[-1] for _ in range(number_of_agents - 1)]
        return tuple([tuple(null_observation), tuple(null_action)])

# TO-DO: father class of DistributedQLearningAgent and LPIAgent with common methods (like auto-update lr) and methods
# to load and save models.

# TO-DO: implement auto-update learning rate
class DistributedQLearningAgent:
    def __init__(self, config):
        self.actions = config['available_actions']
        self.actions_space = len(self.actions)
        self.n_products = config['n_products']

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.eta = config['eta']
        self.tau = config['tau']
        
        self.default_q_value = np.round(-self.n_products / (1 - self.gamma), 3)
        self.values = {}
        self.policy = {}

        self.actions_policy = config['actions_policy']
        self.exploration_prob = config['exploration_prob']

    def update_values(self, observation, action, reward, agents_information):
        action = action - self.actions[0]
    
        if observation not in self.values.keys():
            self.values[observation] = {}
            self.values[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values[observation]['T'] = [0] * self.actions_space

            # in that case the observation must be added also in the policy
            self.policy[observation] = 1/self.actions_space * np.ones(self.actions_space)

        lr = self.alpha

        current_q_value = self.values[observation]['Q'][action]
        next_q_value = (1 - lr) * current_q_value + lr * (reward + self.gamma * agents_information)

        self.values[observation]['Q'][action] = np.round(next_q_value, 3)
        self.values[observation]['T'][action] += 1

    def policy_improvement(self):
        for state in self.policy.keys():
            curr_value = self.policy[state]
            #print(curr_value)
            #print(self.values[state])
            new_value = [np.round(((curr_value[i] ** (1 - self.eta * self.tau)) / np.sum(curr_value)) * np.exp(self.eta * self.values[state]['Q'][i]), 3) for i in range(len(curr_value))]
            self.policy[state] = new_value

    # TO-DO: if remains the same for both algorithms move in father class
    def select_action(self, observation, mask):
        allowed_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        decreased_actions = [action - self.actions[0] for action in allowed_actions]
        
        if self.actions_policy == 'eps-greedy':
            if np.random.rand() < self.exploration_prob:
                print('random action choosen')
                return self.get_random_action(allowed_actions)
            else:
                if observation not in self.values.keys():
                    print('new observation, random action selection')
                    return self.get_random_action(allowed_actions)
                
                q_values = [self.values[observation]['Q'][a] for a in decreased_actions]
                print(f'actions: {allowed_actions}, q_values: {q_values}')
                return allowed_actions[np.argmax(q_values)]
            
        elif self.actions_policy == 'softmax':
            if observation not in self.policy.keys():
                print('new observation, random action selection')
                return self.get_random_action(allowed_actions)
            
            # IMPORTANT: we are computing probability referred only to allowed actions
            policy_values = [self.policy[observation][a] for a in decreased_actions]
            prob_values = policy_values / np.sum(policy_values)
            print(f'actions: {allowed_actions}, prob_values: {prob_values}')
            return np.random.choice(allowed_actions, p=prob_values)

    def get_random_action(self, allowed_actions):
        return np.random.choice(allowed_actions)

    def get_max_value(self, observation):
        if observation not in self.values.keys():
            return self.default_q_value
        return np.max(self.values[observation]['Q'])

# TO-DO: implement auto-update learning rate
class LPIAgent:
    def __init__(self, config, agents_manager, agent_num):
        self.agents_manager = agents_manager
        self.agent_num = agent_num
        self.actions = config['available_actions']
        self.defer = self.actions[-1]
        self.actions_space = len(config['available_actions'])
        self.actions_policy = config['actions_policy']

        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.beta = config['beta']
        self.kappa = config['kappa']
        self.eta = config['eta']
        self.tau = config['tau']
        self.rho = config['rho']
        
        self.n_products = config['n_products']
        self.default_q_value = np.round(-self.n_products/(1 - self.gamma), 3)

        self.values = {}
        self.policy = {}

        self.actions_policy = config['actions_policy']
        self.exploration_prob = config['exploration_prob']

        self.neighbours_kappa = self.agents_manager.get_n_hop_neighbours(self.agent_num, self.kappa)
        self.neighbours_beta = self.agents_manager.get_n_hop_neighbours(self.agent_num, self.beta)

    # IMPORTANT: obs_prime and a_prime contains the observation of the same agent in his next turn and the action
    # that he will choose in that turn (so not next state in general but next state in which the agent has
    # again a product)
    def update_values(self, observation, action, reward, obs_prime, a_prime):
        lr = self.alpha

        action = action - self.actions[0]
        a_prime = a_prime - self.actions[0]

        if observation not in self.values.keys():
            self.values[observation] = {}
            self.values[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values[observation]['T'] = [0] * self.actions_space

            # in that case the observation must be added also in the policy
            self.policy[observation] = 1/self.actions_space * np.ones(self.actions_space)

        curr_value = self.values[observation]['Q'][action]
        # agents_information=q_value dell'azione (a_prime) che l'agente stesso (lui stesso non il successivo) giocherebbe 
        #nello stato s_prime (prossimo stato in cui l'agente ha un prodotto)
        agents_information = self.get_values(obs_prime)[a_prime]

        next_value = (1 - lr) * curr_value + lr * (reward + self.gamma * agents_information)

        self.values[observation]['Q'][action] = np.round(next_value, 3)
        self.values[observation]['T'][action] += 1

    def policy_improvement(self):
        for state in self.policy.keys():
            curr_value = self.policy[state]
            new_value = [np.round(((curr_value[i] ** (1 - self.eta * self.tau)) / np.sum(curr_value)) * np.exp(self.eta * self.values[state]['Q'][i]), 3) for i in range(len(curr_value))]
            self.policy[state] = new_value
            #print(f'curr_value: {curr_value}, new_value: {new_value}')

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
    
    # TO-DO: if remains the same for both algorithms move in father class
    def select_action(self, observation, mask):
        allowed_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        decreased_actions = [action - self.actions[0] for action in allowed_actions]
        
        if self.actions_policy == 'eps-greedy':
            if np.random.rand() < self.exploration_prob:
                print('random action choosen')
                return self.get_random_action(allowed_actions)
            else:
                if observation not in self.values.keys():
                    print('new observation, random action selection')
                    return self.get_random_action(allowed_actions)
                
                q_values = [self.values[observation]['Q'][a] for a in decreased_actions]
                print(f'actions: {allowed_actions}, q_values: {q_values}')
                return allowed_actions[np.argmax(q_values)]
            
        elif self.actions_policy == 'softmax':
            if observation not in self.policy.keys():
                print('new observation, random action selection')
                return self.get_random_action(allowed_actions)
            
            # IMPORTANT: we are computing probability referred only to allowed actions
            policy_values = [self.policy[observation][a] for a in decreased_actions]
            prob_values = policy_values / np.sum(policy_values)
            print(f'actions: {allowed_actions}, prob_values: {prob_values}')
            return np.random.choice(allowed_actions, p=prob_values)
        
    def get_random_action(self, allowed_actions):
        return np.random.choice(allowed_actions)
    
    def get_values(self, observation):
        if observation not in self.values.keys():
            return [self.default_q_value] * self.actions_space
        return self.values[observation]['Q']
    

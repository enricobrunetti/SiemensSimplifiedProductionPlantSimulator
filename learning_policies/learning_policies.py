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
    def __init__(self, actions, n_products, alpha=0.7, gamma=0.9):
        self.actions = actions
        self.actions_space = len(self.actions)
        # learning rate
        self.alpha = alpha
        # discount factor
        self.gamma = gamma
        self.default_q_value = -n_products / (1 - self.gamma)
        self.values = {}

    def update_values(self, observation, action, reward, agents_information):
        action = action - self.actions[0]
    
        if observation not in self.values.keys():
            self.values[observation] = {}
            self.values[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values[observation]['T'] = [0] * self.actions_space

        lr = self.alpha

        current_q_value = self.values[observation]['Q'][action]
        next_q_value = (1 - lr) * current_q_value + lr * (reward + self.gamma * agents_information)

        self.values[observation]['Q'][action] = next_q_value
        self.values[observation]['T'][action] += 1

    def select_action(self, observation, exploration_prob, mask):
        allowed_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        if np.random.rand() < exploration_prob:
            print('random action choosen')
            return np.random.choice(allowed_actions)
        else:
            decreased_actions = [action - self.actions[0] for action in allowed_actions]
            q_values = [self.values[observation]['Q'][a] for a in decreased_actions]
            print(f'actions: {allowed_actions}, q_values: {q_values}')
            return allowed_actions[np.argmax(q_values)]
        
    def get_max_value(self, observation):
        if observation not in self.values.keys():
            return self.default_q_value
        return np.max(self.values[observation]['Q'])

    def get_values(self, observation, action):
        return self.values[observation]['Q'][action]

# TO-DO: implement autp-update lr
class LPIAgent:
    def __init__(self, config, agents_manager, agent_num):
        self.agents_manager = agents_manager
        self.agent_num = agent_num
        self.actions = config['available_actions']
        self.actions_space = len(self.actions)

        # learning rate
        self.alpha = config['alpha']
        # discount factor
        self.gamma = config['gamma']
        self.beta = config['beta']
        self.kappa = config['kappa']

        self.eta = config["eta"]
        self.tau = config["tau"]
        self.rho = config["rho"]

        self.n_products = config['n_products']
        self.default_q_value = -self.n_products / (1 - self.gamma)

        # Q-table
        self.values = {}

        # policy
        self.policy = {}

        self.neighbours_kappa = self.get_n_hop_neighbours(self.agent_num, self.kappa)
        self.neighbours_beta = self.get_n_hop_neighbours(self.agent_num, self.beta)

        self.observation_space_kappa = {}
        self.action_space_kappa = {}

        for agent_kappa in self.neighbours_kappa:
            self.observation_space_kappa[agent_kappa] = set()
            self.action_space_kappa[agent_kappa] = set()

        self.default_global_state = self.agents_manager.get_default_observation(len(self.neighbours_beta))

    def get_max_value(self, agent_num, observation):
        values = self.agents_manager.get_max_value(agent_num, observation)
        if values == None:
            return self.default_q_value
        return values
    
    def get_values(self, agent_num, observation):
        values = self.agents_manager.get_values(agent_num, observation)
        if values == None:
            return [self.default_q_value] * self.actions_space
        return values
    
    def get_q_values_per_action(self, agent_num, observation):        
        neighbor_big_Q = []

        X_index = None
        for action_index in range(len(observation[1])):
            if observation[1][action_index] == "X":
                X_index = action_index
        
        for action_i in self.actions:

            if X_index is not None:
                obs = observation.copy()
                obs[1][X_index] = action_i
                obs = tuple([tuple(obs[0]), tuple(obs[1])])
            else:
                obs = observation

            row_values = self.get_values(agent_num, obs)            
            neighbor_big_Q.append(float(np.mean(row_values)))

        return neighbor_big_Q

    def compute_weighted_log_pi(self, obs_kappa):
        if obs_kappa not in self.policy.keys():
            self.policy[obs_kappa] = 1/self.actions_space * np.ones((self.actions_space))
        w_logpi = self.policy[obs_kappa]
        w_logpi *= np.log([abs(w_logpi[i] + abs(1 / self.actions_space) + 0.1) for i in range(len(w_logpi))]) 
        return list(w_logpi)
    
    def one_policy_improvement(self, observation):
        states = observation[0]
        actions = observation[1]

        big_Q = []

        for agent_kappa in self.neighbours_kappa:
            if agent_kappa == self.agent_num:

                observation_beta = []
                action_beta = []

                for a in self.neighbours_kappa:
                    if a in self.neighbours_beta:
                        observation_beta.append(states[a])
                        if a != agent_kappa:
                            action_beta.append(actions[a])
                print(tuple([tuple(observation_beta), tuple(action_beta)]))
                big_Q.append(self.get_values(self.agent_num, tuple([tuple(observation_beta), tuple(action_beta)])))
                continue

            agent_kappa_neighbors_beta = self.get_n_hop_neighbours(agent_kappa, self.beta)
            agent_kappa_neighbors_beta.sort()

            if len([agent_num for agent_num in agent_kappa_neighbors_beta if agent_num in self.neighbours_kappa]) == len(agent_kappa_neighbors_beta):
                observation_kappa_beta = []
                action_kappa_beta = []
                for agent_num in self.neighbours_kappa:
                    if agent_num in agent_kappa_neighbors_beta:
                        observation_kappa_beta.append(states[agent_num])
                        if agent_num == self.agent_num:
                            action_kappa_beta.append("X")
                        elif agent_num != agent_kappa:
                            action_kappa_beta.append(actions[agent_num])

                global_observation = [observation_kappa_beta, action_kappa_beta]
            else:
                global_observation = self.default_global_state

            big_Q.append(self.get_q_values_per_action(agent_kappa, global_observation))

        big_Q = np.mean(big_Q, axis=0)

        if observation not in self.policy.keys():
            self.policy[observation] = [1/self.action_space] * self.action_space

        self.policy[observation] = \
            [self.policy[observation][i] ** (1 - self.eta * self.tau) * np.exp(self.eta * big_Q[i]) \
            for i in range(len(self.policy[observation]))]
        self.policy[observation] = self.policy[observation] / np.sum(self.policy[observation])

    def update_values(self, observation, action, reward, agents_information):
        action = action - self.actions[0]
        # update the reward using entropy
        reward -= self.eta * self.tau * self.compute_weighted_log_pi(observation)[action]

        if observation not in self.values.keys():
            self.values[observation] = {}
            self.values[observation]['Q'] = [self.default_q_value] * self.actions_space
            self.values[observation]['T'] = [0] * self.actions_space

        lr = self.alpha

        current_q_value = self.values[observation]['Q'][action]
        next_q_value = (1 - lr) * current_q_value + lr * (reward + self.gamma * agents_information)

        self.values[observation]['Q'][action] = next_q_value
        self.values[observation]['T'][action] += 1

    def update_values_with_entropy(self):
        # at the end of the episode, update all the values in the Q table using the entropy
        for key in list(self.values.keys()):
            self.values[key]["Q"] = [self.values[key]["Q"][i] + \
                                             self.eta * self.tau * self.compute_weighted_log_pi(key)[i] \
                                                for i in range(len(self.values[key]["Q"]))]
            
    def end_episode(self, episode_id, product_info, checkpoint, out_dir):

        # add the update of the values with the entopy
        self.update_values_with_entropy()

        # TO-DO: decide to evenutally print statistics about the updated policy

    # TO-DO: add also the softmax action
    def select_action(self, observation, exploration_prob, mask):
        allowed_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        if np.random.rand() < exploration_prob:
            print('random action choosen')
            return np.random.choice(allowed_actions)
        else:
            decreased_actions = [action - self.actions[0] for action in allowed_actions]
            q_values = [self.values[observation]['Q'][a] for a in decreased_actions]
            print(f'actions: {allowed_actions}, q_values: {q_values}')
            return allowed_actions[np.argmax(q_values)]

    def get_n_hop_neighbours(self, agent_num, n_hop):
        return self.agents_manager.get_n_hop_neighbours(agent_num, n_hop)
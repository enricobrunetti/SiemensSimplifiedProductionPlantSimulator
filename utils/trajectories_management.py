import json
import numpy as np

class TrajectoryManager():
    def __init__(self, INPUT_DIR, config):
        self.INPUT_DIR = INPUT_DIR
        # TO-DO: remove, now output dir is different from input only for debug purposes
        self.OUTPUT_DIR = 'output/export_trajectories2_POSTPROCESSED.json'
        self.config = config
        self.n_agents = self.config['n_agents']
        self.n_products = self.config['n_products']
        self.agents_connections = {int(k): v for k, v in self.config['agents_connections'].items()} 

        # load the trajectories
        with open(self.INPUT_DIR, 'r') as infile:
            self.trajectories = json.load(infile)

    # select function to use in order to compute rewards
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_reward(self, reward_type = 'one-step'):
        if reward_type == 'one_step':
            self.compute_one_step_reward()
        elif reward_type == 'semi-mdp':
            self.compute_semiMDP_reward()

    # give the reward of the action performed to the agent that moved there the product
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_one_step_reward(self):
        for episode in self.trajectories:
            for i in range(len(self.trajectories[episode]) - 1):
                self.trajectories[episode][i]['reward'] = self.trajectories[episode][i + 1]['reward'] 

    # give the cumulative reward of all the actions performed until the agent take again the product (if the agent
    # don't take again the product use the comulative reward until the end of the episode)
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def compute_semiMDP_reward(self):
        for episode in self.trajectories:
            for i in range(len(self.trajectories[episode]) - 1):
                agent = self.trajectories[episode][i]['agent']
                reward = self.trajectories[episode][i]['reward']
                j = i + 1
                while j < len(self.trajectories[episode]) and self.trajectories[episode][j]['agent'] != agent:
                    reward += self.trajectories[episode][j]['reward']
                    j += 1
                self.trajectories[episode][i]['reward'] = reward


    # remove from trajectories all the skills
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def remove_production_skill_trajectories(self):
        for episode in self.trajectories:
            self.trajectories[episode] = [step for step in self.trajectories[episode] if step['action'] >= self.config['n_production_skills']]

    # remove from trajectories all the action masks
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def remove_action_masks(self):
        for episode in self.trajectories:
            for step in self.trajectories[episode]:
                del step['state']['action_mask']

    # this function allow to hide states of agents with a distance greater than observability grade for each agent
    # CAUTION: perform this before converting global trajectory into agents trajectories
    def set_states_observability(self, observability_grade):
        agents_state_mask = self.compute_agents_state_mask(observability_grade)
        for episode in self.trajectories:
            for step in self.trajectories[episode]:
                step['state']['agents_state'] = [lst for lst, mask in zip(step['state']['agents_state'], agents_state_mask[step['agent']]) if any(x != 0 for x in mask)]
                products_mask = np.zeros(self.n_products)
                for agent_state in step['state']['agents_state']:
                    if max(agent_state) == 1:
                        products_mask[np.argmax(agent_state)] = 1
                for i in range(len(step['state']['products_state'])):
                    if products_mask[i] == 0:
                        step['state']['products_state'][i] = np.zeros_like(step['state']['products_state'][i]).tolist()

    # compute a mask for each agent based on the observability grade (number of consequent neighbours to have observability on)
    def compute_agents_state_mask(self, observability_grade):
        agents_state_mask = {}
        for agent in range(self.n_agents):
            new_state = []
            for i in range(self.n_agents):
                new_state.append(np.zeros(self.n_products))
            new_state[agent] = np.ones_like(new_state[agent])

            neighbour = {}
            for i in range(observability_grade):
                if i == 0:
                    neighbour = {elem for elem in self.agents_connections[agent] if elem != None}
                else:
                    for elem in neighbour:
                        new_neighbours = {new_elem for new_elem in self.agents_connections[elem] if new_elem != None}
                        neighbour = neighbour.union(new_neighbours)

            for elem in neighbour:
                new_state[elem] = np.ones_like(new_state[elem])
                
            
            agents_state_mask[agent] = np.array(new_state)
        
        return agents_state_mask
    
    # convert global trajectory into agents trajectories
    def extract_agent_trajectories(self):
        filtered_episodes = {}
        for agent in range(self.n_agents):
            agent_name = f'Agent: {agent}'
            filtered_episodes[agent_name] = {}
            for key, value in self.trajectories.items():
                filtered_episodes[agent_name][key] = [filtered_val for filtered_val in value if filtered_val['agent'] == agent]

        self.trajectories = filtered_episodes

    # save as output the trajectory
    def save_trajectory(self):
        with open(self.OUTPUT_DIR, 'w') as outfile:
            json.dump(self.trajectories, outfile, indent=6)

# TO-DO: include the usage of INPUT_DIR
def split_data_single_agent(INPUT_DIR, agent):
    with open('output/export_trajectories2_POSTPROCESSED.json', 'r') as infile:
        trajectories = json.load(infile)
    
    trajectory = trajectories[f'Agent: {agent}']
    t = []
    s = []
    a = []
    r = []
    s_prime = []
    absorbing = []
    sa = []
    for episode in trajectory:
        for i in range(len(trajectory[episode])):
            t.append(trajectory[episode][i]['time'])
            s.append(flatten_dict_values(trajectory[episode][i]['state']))
            a.append(trajectory[episode][i]['action'])
            r.append(trajectory[episode][i]['reward'])
            absorbing.append(1 if i == (len(trajectory[episode]) - 1) else 0)
            temp_sa = flatten_dict_values(trajectory[episode][i]['state'])
            temp_sa.append(trajectory[episode][i]['action'])
            sa.append(temp_sa)

    # TO-DO: fix s_prime
    s_prime = s[1:]
    s_prime.append(s[-1])
    print(len(s))
    print(len(sa))
    print(f'len of s: {len(s[0])}')
    return np.array(t), np.array(s), np.array(a), np.array(r), np.array(s_prime), np.array(absorbing), np.array(sa)

def flatten_dict_values(d):
    flattened_values = []
    if isinstance(d, dict):
        for value in d.values():
            flattened_values.extend(flatten_dict_values(value))
    elif isinstance(d, list):
        for item in d:
            flattened_values.extend(flatten_dict_values(item))
    else:
        flattened_values.append(d)
    return flattened_values



    



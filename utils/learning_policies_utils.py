import numpy as np

def get_agents_informations(config, learning_agents, n_current_agent, observability_grade, state, action):
    n_agents = config['n_agents']
    n_products = config['n_products']
    agents_connections = {int(k): v for k, v in config['agents_connections'].items()}
    neighbours_mask = compute_neighbours_mask(n_agents, n_products, agents_connections, observability_grade)
    #print(neighbours_mask)
    agents_informations = []
    for i in range(n_agents):
        if neighbours_mask[n_current_agent][i][0] == 1 and i != n_current_agent:
            agents_informations.append(learning_agents[i].get_q_value(state, action))
    return np.array(agents_informations)

def compute_neighbours_mask(n_agents, n_products, agents_connections, observability_grade):
        neighbours_mask = {}
        for agent in range(n_agents):
            new_state = []
            for i in range(n_agents):
                new_state.append(np.zeros(n_products))
            new_state[agent] = np.ones_like(new_state[agent])

            neighbour = {}
            for i in range(observability_grade):
                if i == 0:
                    neighbour = {elem for elem in agents_connections[agent] if elem != None}
                else:
                    for elem in neighbour:
                        new_neighbours = {new_elem for new_elem in agents_connections[elem] if new_elem != None}
                        neighbour = neighbour.union(new_neighbours)

            for elem in neighbour:
                new_state[elem] = np.ones_like(new_state[elem])
                
            neighbours_mask[agent] = np.array(new_state)
        
        return neighbours_mask
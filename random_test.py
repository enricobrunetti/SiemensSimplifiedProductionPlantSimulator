import numpy as np
import random
                  
n_agents = 5
n_products = 4
n_observability = 1

#TO-DO: move agents_connections in a config
agents_connections = {0: [None, 1, 2, None],
                        1: [None, 4, 2, 0],
                        2: [1, 3, None, 0],
                        3: [4, None, None, 2],
                        4: [None, None, 3, 1]}

agents_state_mask = {}
for agent in range(n_agents):
    new_state = []
    for i in range(n_agents):
        new_state.append(np.zeros(n_products))
    new_state[agent] = np.ones_like(new_state[agent])

    neighbour = {}
    print(neighbour)
    for i in range(n_observability):
        if i == 0:
            neighbour = {elem for elem in agents_connections[agent] if elem != None}
        else:
            for elem in neighbour:
                new_neighbours = {new_elem for new_elem in agents_connections[elem] if new_elem != None}
                neighbour = neighbour.union(new_neighbours)
        print(neighbour)

    for elem in neighbour:
        new_state[elem] = np.ones_like(new_state[elem])
        
    
    agents_state_mask[agent] = np.array(new_state)

print(agents_state_mask)
    
import json
import numpy as np

# TO-DO: add the confing and properly set up INPUT_DIR and n_products
def extract_agent_trajectories(INPUT_DIR, n_agents, n_products):

    with open('output/export_trajectories.json', 'r') as infile:
        trajectories = json.load(infile)
        filtered_episodes = {}
        for agent in range(n_agents):
            agent_name = f'Agent: {agent}'
            filtered_episodes[agent_name] = {}
            for key, value in trajectories.items():
                filtered_episodes[agent_name][key] = [filtered_val for filtered_val in value if filtered_val['agent'] == agent]

    with open('output/test.json', 'w') as outfile:
        json.dump(filtered_episodes, outfile, indent=6)

# This function allow to set the number of consequent neighbours on which each agent have visibility
# TO-DO: properly set up config and in particular setup INPUT_DIR
def set_agents_state_observability(INPUT_DIR, n_agents, n_products, observability_grade):
    agents_state_mask = compute_agents_state_mask(n_agents, n_products, observability_grade)
    with open('output/test.json', 'r') as infile:
        trajectories = json.load(infile)
        for agent in trajectories:
            for episode in trajectories[agent]:
                for elem in trajectories[agent][episode]:
                    for i in range(n_agents):
                        for j in range(n_products):
                            if agents_state_mask[int(agent[-1])][i][j] == 0:
                                elem['state']['agents_state'][i][j] = None

    with open('output/filtered_observability_test.json', 'w') as outfile:
        json.dump(trajectories, outfile, indent=6)



def compute_agents_state_mask(n_agents, n_products, observability_grade):
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
        for i in range(observability_grade):
            if i == 0:
                neighbour = {elem for elem in agents_connections[agent] if elem != None}
            else:
                for elem in neighbour:
                    new_neighbours = {new_elem for new_elem in agents_connections[elem] if new_elem != None}
                    neighbour = neighbour.union(new_neighbours)

        for elem in neighbour:
            new_state[elem] = np.ones_like(new_state[elem])
            
        
        agents_state_mask[agent] = np.array(new_state)
    
    return agents_state_mask
    



import json

# TO-DO: add the possibility to limit the state visibility of every signle agent
def extract_agent_trajectories(INPUT_DIR, n_agents):
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


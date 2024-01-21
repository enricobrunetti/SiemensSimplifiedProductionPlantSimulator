from learning_policies.distributed_qlearning import QLearningAgent
from utils.trajectories_management import split_data_single_agent

INPUT_DIR = "output/export_trajectories6_POSTPROCESSED.json"

num_agents = 9
actions = [6, 7, 8, 9, 10]

agents_qlearning = [QLearningAgent(actions) for _ in range(num_agents)]

for i in range(num_agents):
    agent = agents_qlearning[i]
    _, states, actions, rewards, s_prime, absorbing, _, _ = split_data_single_agent(INPUT_DIR, i)

    for j in range(len(states)):
        agent.update_q_value(states[j], actions[j], rewards[j], s_prime[j])

_, states, _, _, _, _, _, masks = split_data_single_agent(INPUT_DIR, 0)
selected_action = agents_qlearning[0].select_action(states[0], 0.5, masks[0])
print(f'action {selected_action} has been choosen for state {states[1]}')
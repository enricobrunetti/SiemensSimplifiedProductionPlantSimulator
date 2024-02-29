import matplotlib.pyplot as plt
import numpy as np

class RewardVisualizer:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        n_rows = int(np.ceil(n_agents / 3))
        self.fig, self.ax = plt.subplots(n_rows, 3, figsize=(15, 6 * n_rows))
        self.ax = np.array(self.ax).flatten()
        for i in range(n_agents):
            self.ax[i].set_xlabel('Episode')
            self.ax[i].set_ylabel('Reward')
            self.ax[i].set_title(f'Agent {i+1}')
        self.lines = [ax.plot([], [], 'o-', label=f'Agent {i+1}')[0] for i, ax in enumerate(self.ax)]
        self.rewards = [[] for _ in range(n_agents)]
        self.time_steps = [[] for _ in range(n_agents)]
        plt.tight_layout()

    def update_plot(self, agent_id, time_step, reward):
        self.time_steps[agent_id].append(time_step)
        self.rewards[agent_id].append(reward)
        self.lines[agent_id].set_data(self.time_steps[agent_id], self.rewards[agent_id])
        self.ax[agent_id].relim()
        self.ax[agent_id].autoscale_view()

    def show_plot(self):
        plt.show()

    def save_plot(self, filename):
        plt.savefig(filename)
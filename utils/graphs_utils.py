import matplotlib.pyplot as plt
import numpy as np
import json

class DistQAndLPIPlotter:
    def __init__(self, model_path, baseline_path):
        self.model_path = model_path
        self.baseline_path = baseline_path

        with open(f"{self.model_path}/reward_for_plot_training.json", 'r') as infile:
            self.training_performances = json.load(infile)

        with open(f"{self.model_path}/reward_for_plot_test.json", 'r') as infile:
            self.test_performances = json.load(infile)

        with open(f"{self.baseline_path}/reward_for_plot_test.json", 'r') as infile:
            self.baseline_performances = json.load(infile)

    def plot_single_agent_reward_graph(self, agent_num):
        training_rewards = [episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances.values()]
        test_rewards = [episode["agents_reward_for_plot"][str(agent_num)] for episode in self.test_performances.values()]
        baseline_rewards = [episode["agents_reward_for_plot"][str(agent_num)] for episode in self.baseline_performances.values()]

        # Downsampling with convolution
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size

        training_smoothed_reward = np.convolve(training_rewards, kernel, mode='valid')

        # 95% confidence intervals computation
        training_mean = np.mean(training_smoothed_reward)
        training_std = np.std(training_smoothed_reward)
        training_ci = 1.96 * (training_std / np.sqrt(len(training_smoothed_reward)))

        test_mean = np.mean(test_rewards)
        test_std = np.std(test_rewards)
        test_ci = 1.96 * (test_std / np.sqrt(len(test_rewards)))

        baseline_mean = np.mean(baseline_rewards)
        baseline_std = np.std(baseline_rewards)
        baseline_ci = 1.96 * (baseline_std / np.sqrt(len(baseline_rewards)))

        # Plot creation
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]})
        # Plot for training phase
        axs[0].plot(training_smoothed_reward, label='Training Reward', color='blue')
        axs[0].fill_between(range(len(training_smoothed_reward)), training_smoothed_reward - training_ci, training_smoothed_reward + training_ci, color='orange', alpha=0.5, label='Confidence Interval 95%')
        axs[0].set_xticks([])
        axs[0].set_title('Training Phase')
        axs[0].legend()
        axs[0].set_ylabel('Reward')
        axs[0].set_xlabel('Episodes')

        # Plot for test phase
        axs[1].errorbar(0.4, test_mean, yerr=[test_ci], fmt='o', capsize=5, label='Test Reward', color='green')
        axs[1].errorbar(0.6, baseline_mean, yerr=[baseline_ci], fmt='o', capsize =5, label='Baseline Reward', color='red')
        axs[1].set_xticks([])
        axs[1].set_xlim(0, 1)
        axs[1].set_title('Test Phase')
        axs[1].legend()
        axs[1].set_ylabel('Reward')
        fig.suptitle(f"Agent {agent_num+1}")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_agent{agent_num}_reward.png')

    def plot_reward_graphs(self):
        n_agents = len(self.training_performances['0']['agents_reward_for_plot'])

        for i in range(n_agents):
            self.plot_single_agent_reward_graph(i)

    def plot_performance_graph(self):
        training_duration = [episode["episode_duration"] for episode in self.training_performances.values()]
        test_duration = [episode["episode_duration"] for episode in self.test_performances.values()]
        baseline_duration = [episode["episode_duration"] for episode in self.baseline_performances.values()]

        # Downsampling with convolution
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size

        training_smoothed_duration = np.convolve(training_duration, kernel, mode='valid')

        # 95% confidence intervals computation
        training_mean = np.mean(training_smoothed_duration)
        training_std = np.std(training_smoothed_duration)
        training_ci = 1.96 * (training_std / np.sqrt(len(training_smoothed_duration)))

        test_mean = np.mean(test_duration)
        test_std = np.std(test_duration)
        test_ci = 1.96 * (test_std / np.sqrt(len(test_duration)))

        baseline_mean = np.mean(baseline_duration)
        baseline_std = np.std(baseline_duration)
        baseline_ci = 1.96 * (baseline_std / np.sqrt(len(baseline_duration)))

        # Plot creation
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]})

        # Plot for training phase
        axs[0].plot(training_smoothed_duration, label='Training Duration', color='blue')
        axs[0].fill_between(range(len(training_smoothed_duration)), training_smoothed_duration - training_ci, training_smoothed_duration + training_ci, color='orange', alpha=0.5, label='Confidence Interval 95%')
        axs[0].set_xticks([])
        axs[0].set_title('Training Phase')
        axs[0].legend()
        axs[0].set_ylabel('Episodes Duration')
        axs[0].set_xlabel('Episodes')

        # Plot for test phase
        axs[1].errorbar(0.4, test_mean, yerr=[test_ci], fmt='o', capsize=5, label='Test Duration', color='green')
        axs[1].errorbar(0.6, baseline_mean, yerr=[baseline_ci], fmt='o', capsize =5, label='Baseline Duration', color='red')
        axs[1].set_xticks([])
        axs[1].set_xlim(0, 1)
        axs[1].set_title('Test Phase')
        axs[1].legend()
        axs[1].set_ylabel('Episodeds Duration')
        fig.suptitle(f"Duration Performance Plot")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration.png')

# non servirà più
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
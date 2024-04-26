import matplotlib.pyplot as plt
import numpy as np
import json
from bootstrapped import bootstrap as bs
from bootstrapped import stats_functions as bs_stats

class DistQAndLPIPlotter:
    def __init__(self, model_path_runs, baseline_path_runs):
        self.model_path_runs = model_path_runs
        self.model_path = self.model_path_runs[0].rsplit("/", 1)[0]
        self.baseline_path_runs = baseline_path_runs

        self.training_performances = {}
        self.test_performances = {}
        self.baseline_performances = {}
        for path in self.model_path_runs.values():
            with open(f"{path}/reward_for_plot_training.json", 'r') as infile:
                self.training_performances[path] = json.load(infile)

            with open(f"{path}/reward_for_plot_test.json", 'r') as infile:
                self.test_performances[path] = json.load(infile)

        for path in self.baseline_path_runs.values():
            with open(f"{path}/reward_for_plot_test.json", 'r') as infile:
                self.baseline_performances[path] = json.load(infile)

    def plot_single_agent_reward_graph(self, agent_num):
        training_rewards = []
        test_rewards = []
        baseline_rewards = []

        for path in self.model_path_runs.values():
            training_rewards.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances[path].values()])
            test_rewards.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.test_performances[path].values()])
        
        for path in self.baseline_path_runs.values():
            baseline_rewards.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.baseline_performances[path].values()])

        training_rewards_all_runs = []
        num_episodes = len(training_rewards[0])
        for i in range(num_episodes):
            training_rewards_all_runs.append([row[i] for row in training_rewards])

        test_rewards_all_runs = np.array(test_rewards).flatten()
        baseline_rewards_all_runs = np.array(baseline_rewards).flatten()

        # 95% confidence intervals computation
        confidence_level = 0.95
        # Number of bootstraps
        num_bootstraps = 1000
        alpha = 1 - confidence_level

        # Compute confidence intervals
        lower_ci_training = []
        mean_training = []
        upper_ci_training = []
        for elem in training_rewards_all_runs:
            res_training_bs = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
            # Lower and upper confidence intervals
            lower_ci_training.append(res_training_bs.lower_bound)
            upper_ci_training.append(res_training_bs.upper_bound)
            mean_training.append(np.mean(elem))
        
        # Compute confidence intervals
        res_test_bs = bs.bootstrap(test_rewards_all_runs, stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
        # Lower and upper confidence intervals
        lower_ci_test = res_test_bs.lower_bound
        upper_ci_test = res_test_bs.upper_bound
        mean_test = np.mean(test_rewards_all_runs)

        # Compute confidence intervals
        res_baseline_bs = bs.bootstrap(baseline_rewards_all_runs, stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
        # Lower and upper confidence intervals
        lower_ci_baseline = res_baseline_bs.lower_bound
        upper_ci_baseline = res_baseline_bs.upper_bound
        mean_baseline = np.mean(baseline_rewards_all_runs)

        # Downsampling with convolution
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size

        training_smoothed_reward_lower = np.convolve(lower_ci_training, kernel, mode='valid')
        training_smoothed_reward = np.convolve(mean_training, kernel, mode='valid')
        training_smoothed_reward_upper = np.convolve(upper_ci_training, kernel, mode='valid')

        # Plot creation
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
        # Plot for training phase
        axs[0].plot(range(len(training_smoothed_reward)), training_smoothed_reward, label='Training Rewards Convolved')
        axs[0].fill_between(range(len(training_smoothed_reward)), training_smoothed_reward_lower, training_smoothed_reward_upper, color='yellow', alpha=0.2, label='Convolved Reward Confidence Interval 95%')
        axs[0].set_title('Training Phase')
        axs[0].legend()
        axs[0].set_ylabel('Reward')
        axs[0].set_xlabel('Episodes')

        # Plot for test phase
        axs[1].errorbar(0.4, mean_test, yerr=[[mean_test - lower_ci_test], [upper_ci_test - mean_test]], fmt='o', capsize=5, label='Test Reward', color='green')
        axs[1].errorbar(0.6, mean_baseline, yerr=[[mean_baseline - lower_ci_baseline], [upper_ci_baseline - mean_baseline]], fmt='o', capsize=5, label='Baseline Reward', color='red')
        axs[1].set_xticks([])
        axs[1].set_xlim(0, 1)
        axs[1].set_title('Test Phase')
        axs[1].legend()
        axs[1].set_ylabel('Reward')
        axs[1].tick_params(left=True, labelleft=True)
        fig.suptitle(f"Agent {agent_num+1}")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_agent{agent_num}_reward.png')

    def plot_reward_graphs(self):
        n_agents = len(self.training_performances[self.model_path_runs[0]]['0']['agents_reward_for_plot'])

        for i in range(n_agents):
            self.plot_single_agent_reward_graph(i)

    def plot_performance_graph(self):
        training_duration = []
        test_duration = []
        baseline_duration = []

        for path in self.model_path_runs.values():
            training_duration.append([episode["episode_duration"] for episode in self.training_performances[path].values()])
            test_duration.append([episode["episode_duration"] for episode in self.test_performances[path].values()])

        for path in self.baseline_path_runs.values():
            baseline_duration.append([episode["episode_duration"] for episode in self.baseline_performances[path].values()])

        training_performance_all_runs = []
        num_episodes = len(training_duration[0])
        for i in range(num_episodes):
            training_performance_all_runs.append([row[i] for row in training_duration])

        test_performance_all_runs = np.array(test_duration).flatten()
        baseline_duration_all_runs = np.array(baseline_duration).flatten()

        # 95% confidence intervals computation
        confidence_level = 0.95
        # Number of bootstraps
        num_bootstraps = 1000
        alpha = 1 - confidence_level

        # Compute confidence intervals
        lower_ci_training = []
        mean_training = []
        upper_ci_training = []
        for elem in training_performance_all_runs:
            res_training_bs = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
            # Lower and upper confidence intervals
            lower_ci_training.append(res_training_bs.lower_bound)
            upper_ci_training.append(res_training_bs.upper_bound)
            mean_training.append(np.mean(elem))
        
        # Compute confidence intervals
        res_test_bs = bs.bootstrap(test_performance_all_runs, stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
        # Lower and upper confidence intervals
        lower_ci_test = res_test_bs.lower_bound
        upper_ci_test = res_test_bs.upper_bound
        mean_test = np.mean(test_performance_all_runs)

        # Compute confidence intervals
        res_baseline_bs = bs.bootstrap(baseline_duration_all_runs, stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
        # Lower and upper confidence intervals
        lower_ci_baseline = res_baseline_bs.lower_bound
        upper_ci_baseline = res_baseline_bs.upper_bound
        mean_baseline = np.mean(baseline_duration_all_runs)

        # Downsampling with convolution
        kernel_size = 100
        kernel = np.ones(kernel_size) / kernel_size

        training_smoothed_duration_lower = np.convolve(lower_ci_training, kernel, mode='valid')
        training_smoothed_duration = np.convolve(mean_training, kernel, mode='valid')
        training_smoothed_duration_upper = np.convolve(upper_ci_training, kernel, mode='valid')

        # Plot creation
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)

        # Plot for training phase
        axs[0].plot(range(len(training_smoothed_duration)), training_smoothed_duration, label='Training Duration Convolved')
        axs[0].fill_between(range(len(training_smoothed_duration)), training_smoothed_duration_lower, training_smoothed_duration_upper, color='yellow', alpha=0.2, label='Convolved Duration Confidence Interval 95%')
        axs[0].set_title('Training Phase')
        axs[0].legend()
        axs[0].set_ylabel('Episodes Duration')
        axs[0].set_xlabel('Episodes')

        # Plot for test phase
        axs[1].errorbar(0.4, mean_test, yerr=[[mean_test - lower_ci_test], [upper_ci_test - mean_test]], fmt='o', capsize=5, label='Test Duration', color='green')
        axs[1].errorbar(0.6, mean_baseline, yerr=[[mean_baseline- lower_ci_baseline], [upper_ci_baseline - mean_baseline]], fmt='o', capsize =5, label='Baseline Duration', color='red')
        axs[1].set_xticks([])
        axs[1].set_xlim(0, 1)
        axs[1].set_title('Test Phase')
        axs[1].legend()
        axs[1].set_ylabel('Episodeds Duration')
        axs[1].tick_params(left=True, labelleft=True)
        fig.suptitle(f"Duration Performance Plot")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration.png')
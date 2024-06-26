import matplotlib.pyplot as plt
import numpy as np
import json
from bootstrapped import bootstrap as bs
from bootstrapped import stats_functions as bs_stats

class DistQAndLPIPlotter:
    def __init__(self, model_path_runs, baseline_path_runs, n_episodes):
        self.model_path_runs = model_path_runs
        self.model_path = self.model_path_runs[0].rsplit("/", 1)[0]
        self.baseline_path_runs = baseline_path_runs
        self.kernel_size = int(n_episodes / 10)

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
        kernel = np.ones(self.kernel_size) / self.kernel_size

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
        fig.suptitle(f"Agent {agent_num+1} Reward")
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
        kernel = np.ones(self.kernel_size) / self.kernel_size

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
        fig.suptitle(f"Duration Performance")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration.png')

class FQIPlotter:
    def __init__(self, model_path_runs, test_episodes_for_fqi_iteration, multiple_exploration_probabilities, exploration_probabilities, episodes_for_each_explor_for_iteration):
        self.model_path_runs = model_path_runs
        self.model_path = self.model_path_runs[0].rsplit("/", 1)[0]
        self.kernel_size = 2#int(n_episodes / 10)
        self.test_episodes_for_fqi_iteration = test_episodes_for_fqi_iteration
        self.multiple_exploration_probabilities = multiple_exploration_probabilities
        self.exploration_probabilities = exploration_probabilities
        self.episodes_for_each_explor_for_iteration = episodes_for_each_explor_for_iteration

        self.training_performances = {}
        self.training_performances_greedy = {}
        for path in self.model_path_runs.values():
            with open(f"{path}/reward_for_plot_training.json", 'r') as infile:
                self.training_performances[path] = json.load(infile)
            with open(f"{path}/reward_for_greedy_plot_training.json", 'r') as infile:
                self.training_performances_greedy[path] = json.load(infile)

    def plot_single_agent_reward_graph(self, agent_num):
        training_rewards = []
        training_rewards_greedy = []

        for path in self.model_path_runs.values():
            training_rewards.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances[path].values()])
            training_rewards_greedy.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances_greedy[path].values()])

        training_rewards = self.compute_mean(training_rewards, self.test_episodes_for_fqi_iteration - 1)

        training_rewards_all_runs = []
        num_episodes = len(training_rewards[0])
        for i in range(num_episodes):
            training_rewards_all_runs.append([row[i] for row in training_rewards])

        training_rewards_all_runs_greedy = []
        num_episodes = len(training_rewards_greedy[0])
        for i in range(num_episodes):
            training_rewards_all_runs_greedy.append([row[i] for row in training_rewards_greedy])

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

        lower_ci_training_greedy = []
        mean_training_greedy = []
        upper_ci_training_greedy = []
        for elem in training_rewards_all_runs_greedy:
            res_training_bs_greedy = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
            # Lower and upper confidence intervals
            lower_ci_training_greedy.append(res_training_bs_greedy.lower_bound)
            upper_ci_training_greedy.append(res_training_bs_greedy.upper_bound)
            mean_training_greedy.append(np.mean(elem))

        # Downsampling with convolution
        kernel = np.ones(self.kernel_size) / self.kernel_size

        training_smoothed_reward_lower = np.convolve(lower_ci_training, kernel, mode='valid')
        training_smoothed_reward = np.convolve(mean_training, kernel, mode='valid')
        training_smoothed_reward_upper = np.convolve(upper_ci_training, kernel, mode='valid')

        training_smoothed_reward_lower_greedy = np.convolve(lower_ci_training_greedy, kernel, mode='valid')
        training_smoothed_reward_greedy = np.convolve(mean_training_greedy, kernel, mode='valid')
        training_smoothed_reward_upper_greedy = np.convolve(upper_ci_training_greedy, kernel, mode='valid')

        # Plot creation
        fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        # Plot for training phase
        axs1.plot(range(len(training_smoothed_reward)), training_smoothed_reward, label='Training Rewards Convolved')
        axs1.fill_between(range(len(training_smoothed_reward)), training_smoothed_reward_lower, training_smoothed_reward_upper, color='yellow', alpha=0.2, label='Convolved Reward Confidence Interval 95%')
        axs1.set_title('Training Phase Eps-Greedy')
        axs1.legend()
        axs1.set_ylabel('Reward')
        axs2.plot(range(len(training_smoothed_reward_greedy)), training_smoothed_reward_greedy, label='Training Rewards Convolved')
        axs2.fill_between(range(len(training_smoothed_reward_greedy)), training_smoothed_reward_lower_greedy, training_smoothed_reward_upper_greedy, color='yellow', alpha=0.2, label='Convolved Reward Confidence Interval 95%')
        axs2.set_title('Training Phase Greedy')
        axs2.legend()
        axs2.set_ylabel('Reward')
        axs2.set_xlabel('Episodes')

        fig.suptitle(f"Agent {agent_num+1} Reward")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_agent{agent_num}_reward.png')

    def plot_single_agent_single_run_reward_graph(self, agent_num):
        training_rewards = []
        training_rewards_greedy = []

        for path in self.model_path_runs.values():
            training_rewards.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances[path].values()])
            training_rewards_greedy.append([episode["agents_reward_for_plot"][str(agent_num)] for episode in self.training_performances_greedy[path].values()])

        training_rewards = self.compute_mean(training_rewards, self.test_episodes_for_fqi_iteration - 1)

        # Downsampling with convolution
        kernel = np.ones(self.kernel_size) / self.kernel_size

        training_reward_convolved = []
        training_reward_convolved_greedy = []

        for single_agent_training in training_rewards:
            training_reward_convolved.append(np.convolve(single_agent_training, kernel, mode='valid'))
        
        for single_agent_training_greedy in training_rewards_greedy:
            training_reward_convolved_greedy.append(np.convolve(single_agent_training_greedy, kernel, mode='valid'))

        # Plot creation
        fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Plot for training phase
        for i in range(len(training_reward_convolved)):
            axs1.plot(range(len(training_reward_convolved[i])), training_reward_convolved[i], label=f'Training Reward Convolved Run {i+1}')
        axs1.set_title('Training Phase Eps-Greedy')
        axs1.legend()
        axs1.set_ylabel('Reward')
        for i in range(len(training_reward_convolved_greedy)):
            axs2.plot(range(len(training_reward_convolved_greedy[i])), training_reward_convolved_greedy[i], label=f'Training Reward Convolved Run {i+1}')
        axs2.set_title('Training Phase Greedy')
        axs2.legend()
        axs2.set_ylabel('Reward')
        axs2.set_xlabel('Episodes')

        fig.suptitle(f"Agent {agent_num+1} Reward Single Runs")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_agent{agent_num}_reward_single_runs.png')

    def plot_reward_graphs(self):
        n_agents = len(self.training_performances[self.model_path_runs[0]]['0']['agents_reward_for_plot'])

        for i in range(n_agents):
            self.plot_single_agent_reward_graph(i)
            self.plot_single_agent_single_run_reward_graph(i)

    def plot_performance_graph(self):
        training_duration = []
        training_duration_greedy = []

        for path in self.model_path_runs.values():
            training_duration.append([episode["episode_duration"] for episode in self.training_performances[path].values()])
            training_duration_greedy.append([episode["episode_duration"] for episode in self.training_performances_greedy[path].values()])

        training_duration = self.compute_mean(training_duration, self.test_episodes_for_fqi_iteration - 1)

        training_performance_all_runs = []
        num_episodes = len(training_duration[0])
        for i in range(num_episodes):
            training_performance_all_runs.append([row[i] for row in training_duration])

        training_performance_all_runs_greedy = []
        num_episodes = len(training_duration_greedy[0])
        for i in range(num_episodes):
            training_performance_all_runs_greedy.append([row[i] for row in training_duration_greedy])

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

        lower_ci_training_greedy = []
        mean_training_greedy = []
        upper_ci_training_greedy = []
        for elem in training_performance_all_runs_greedy:
            res_training_bs_greedy = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
            # Lower and upper confidence intervals
            lower_ci_training_greedy.append(res_training_bs_greedy.lower_bound)
            upper_ci_training_greedy.append(res_training_bs_greedy.upper_bound)
            mean_training_greedy.append(np.mean(elem))

        # Downsampling with convolution
        kernel = np.ones(self.kernel_size) / self.kernel_size

        training_smoothed_duration_lower = np.convolve(lower_ci_training, kernel, mode='valid')
        training_smoothed_duration = np.convolve(mean_training, kernel, mode='valid')
        training_smoothed_duration_upper = np.convolve(upper_ci_training, kernel, mode='valid')

        training_smoothed_duration_lower_greedy = np.convolve(lower_ci_training_greedy, kernel, mode='valid')
        training_smoothed_duration_greedy = np.convolve(mean_training_greedy, kernel, mode='valid')
        training_smoothed_duration_upper_greedy = np.convolve(upper_ci_training_greedy, kernel, mode='valid')

        # Plot creation
        fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Plot for training phase
        axs1.plot(range(len(training_smoothed_duration)), training_smoothed_duration, label='Training Duration Convolved')
        axs1.fill_between(range(len(training_smoothed_duration)), training_smoothed_duration_lower, training_smoothed_duration_upper, color='yellow', alpha=0.2, label='Convolved Duration Confidence Interval 95%')
        axs1.set_title('Training Phase Eps-Greedy')
        axs1.legend()
        axs1.set_ylabel('Episodes Duration')
        axs2.plot(range(len(training_smoothed_duration_greedy)), training_smoothed_duration_greedy, label='Training Duration Convolved')
        axs2.fill_between(range(len(training_smoothed_duration_greedy)), training_smoothed_duration_lower_greedy, training_smoothed_duration_upper_greedy, color='yellow', alpha=0.2, label='Convolved Duration Confidence Interval 95%')
        axs2.set_title('Training Phase Greedy')
        axs2.legend()
        axs2.set_ylabel('Episodes Duration')
        axs2.set_xlabel('Episodes')

        fig.suptitle(f"Duration Performance")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration.png')

    def plot_performance_graph_multiple_epsilon(self):
        training_duration = []
        training_duration_greedy = []

        for path in self.model_path_runs.values():
            training_duration.append([episode["episode_duration"] for episode in self.training_performances[path].values()])
            training_duration_greedy.append([episode["episode_duration"] for episode in self.training_performances_greedy[path].values()])

        training_duration = self.compute_mean(training_duration, self.episodes_for_each_explor_for_iteration)

        training_performance_all_runs = {}
        for j in range(len(self.exploration_probabilities)):
            training_performance_all_runs[j] = []

        num_episodes = int(len(training_duration[0]) / len(self.exploration_probabilities))
        for i in range(num_episodes):
            for j in range(len(self.exploration_probabilities)):
                training_performance_all_runs[j].append([row[i + j] for row in training_duration])

        training_performance_all_runs_greedy = []
        num_episodes = len(training_duration_greedy[0])
        for i in range(num_episodes):
            training_performance_all_runs_greedy.append([row[i] for row in training_duration_greedy])

        # 95% confidence intervals computation
        confidence_level = 0.95
        # Number of bootstraps
        num_bootstraps = 1000
        alpha = 1 - confidence_level

        # Compute confidence intervals
        lower_ci_training = {}
        mean_training = {}
        upper_ci_training = {}
        for j in range(len(self.exploration_probabilities)):
            lower_ci_training[j] = []
            mean_training[j] = []
            upper_ci_training[j] = []
            for elem in training_performance_all_runs[j]:
                res_training_bs = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
                # Lower and upper confidence intervals
                lower_ci_training[j].append(res_training_bs.lower_bound)
                upper_ci_training[j].append(res_training_bs.upper_bound)
                mean_training[j].append(np.mean(elem))

        lower_ci_training_greedy = []
        mean_training_greedy = []
        upper_ci_training_greedy = []
        for elem in training_performance_all_runs_greedy:
            res_training_bs_greedy = bs.bootstrap(np.array(elem), stat_func=bs_stats.mean, num_iterations=num_bootstraps, alpha=alpha)
            # Lower and upper confidence intervals
            lower_ci_training_greedy.append(res_training_bs_greedy.lower_bound)
            upper_ci_training_greedy.append(res_training_bs_greedy.upper_bound)
            mean_training_greedy.append(np.mean(elem))

        # Downsampling with convolution
        kernel = np.ones(self.kernel_size) / self.kernel_size

        training_smoothed_duration_lower = []
        training_smoothed_duration = []
        training_smoothed_duration_upper = []
        for j in range(len(self.exploration_probabilities)):
            training_smoothed_duration_lower.append(np.convolve(lower_ci_training[j], kernel, mode='valid'))
            training_smoothed_duration.append(np.convolve(mean_training[j], kernel, mode='valid'))
            training_smoothed_duration_upper.append(np.convolve(upper_ci_training[j], kernel, mode='valid'))

        training_smoothed_duration_lower_greedy = np.convolve(lower_ci_training_greedy, kernel, mode='valid')
        training_smoothed_duration_greedy = np.convolve(mean_training_greedy, kernel, mode='valid')
        training_smoothed_duration_upper_greedy = np.convolve(upper_ci_training_greedy, kernel, mode='valid')

        # Plot creation
        fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Plot for training phase
        for j in range(len(self.exploration_probabilities)):
            axs1.plot(range(len(training_smoothed_duration[j])), training_smoothed_duration[j], label=f'Training Duration Convolved Eps={self.exploration_probabilities[j]}')
            axs1.fill_between(range(len(training_smoothed_duration[j])), training_smoothed_duration_lower[j], training_smoothed_duration_upper[j], alpha=0.2, label=f'Convolved Duration Confidence Interval 95% Eps={self.exploration_probabilities[j]}')
        axs1.set_title('Training Phase Eps-Greedy')
        axs1.legend()
        axs1.set_ylabel('Episodes Duration')
        axs2.plot(range(len(training_smoothed_duration_greedy)), training_smoothed_duration_greedy, label='Training Duration Convolved')
        axs2.fill_between(range(len(training_smoothed_duration_greedy)), training_smoothed_duration_lower_greedy, training_smoothed_duration_upper_greedy, color='yellow', alpha=0.2, label='Convolved Duration Confidence Interval 95%')
        axs2.set_title('Training Phase Greedy')
        axs2.legend()
        axs2.set_ylabel('Episodes Duration')
        axs2.set_xlabel('Episodes')

        fig.suptitle(f"Duration Performance")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration.png')

    def plot_single_run_performance_graph(self):
        training_duration = []
        training_duration_greedy = []

        for path in self.model_path_runs.values():
            training_duration.append([episode["episode_duration"] for episode in self.training_performances[path].values()])
            training_duration_greedy.append([episode["episode_duration"] for episode in self.training_performances_greedy[path].values()])

        training_duration = self.compute_mean(training_duration, self.test_episodes_for_fqi_iteration - 1)

        training_duration_convolved = []
        training_duration_convolved_greedy = []

        # Downsampling with convolution
        kernel = np.ones(self.kernel_size) / self.kernel_size

        for single_agent_training in training_duration:
            training_duration_convolved.append(np.convolve(single_agent_training, kernel, mode='valid'))

        for single_agent_training_greedy in training_duration_greedy:
            training_duration_convolved_greedy.append(np.convolve(single_agent_training_greedy, kernel, mode='valid'))

        # Plot creation
        fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

        # Plot for training phase
        for i in range(len(training_duration_convolved)):
            axs1.plot(range(len(training_duration_convolved[i])), training_duration_convolved[i], label=f'Training Duration Convolved Run {i+1}')
        axs1.set_title('Training Phase Eps-Greedy')
        axs1.legend()
        axs1.set_ylabel('Episodes Duration')
        for i in range(len(training_duration_convolved_greedy)):
            axs2.plot(range(len(training_duration_convolved_greedy[i])), training_duration_convolved_greedy[i], label=f'Training Duration Convolved Run {i+1}')
        axs2.set_title('Training Phase Greedy')
        axs2.legend()
        axs2.set_ylabel('Episodes Duration')
        axs2.set_xlabel('Episodes')

        fig.suptitle(f"Duration Performance Single Runs")
        plt.tight_layout()
        plt.savefig(f'{self.model_path}/plot_performance_duration_single_runs.png')
    
    def compute_mean(self, input_matrix, n_mean_elem):
        output_matrix = []
        for elem in input_matrix:
            means = []
            for i in range(0, len(elem), n_mean_elem):
                group = elem[i:i+n_mean_elem]
                means_group = sum(group) / float(n_mean_elem)
                means.append(means_group)
            output_matrix.append(means)
        return output_matrix


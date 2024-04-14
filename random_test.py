import numpy as np
import random
import json

import math

import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
episodes_train = np.arange(1, 101)  # Numero di episodi di training
episodes_test = np.arange(1, 21)  # Numero di episodi di test
training_rewards = np.random.normal(0.5, 0.1, size=100)  # Reward di training casuali (sostituisci con i tuoi dati)
test_rewards = np.random.normal(0.6, 0.1, size=20)  # Reward di test casuali (sostituisci con i tuoi dati)
baseline_rewards = np.random.normal(0.4, 0.05, size=20)  # Baseline reward casuali (sostituisci con i tuoi dati)

# Calcolo degli intervalli di confidenza al 95%
training_mean = np.mean(training_rewards)
training_std = np.std(training_rewards)
training_ci = 1.96 * (training_std / np.sqrt(len(training_rewards)))

test_mean = np.mean(test_rewards)
test_std = np.std(test_rewards)
test_ci = 1.96 * (test_std / np.sqrt(len(test_rewards)))

baseline_mean = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)
baseline_ci = 1.96 * (baseline_std / np.sqrt(len(baseline_rewards)))

# Creazione del grafico
fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1]})

# Plot per la fase di training
axs[0].plot(episodes_train, training_rewards, label='Training Reward', color='blue')
axs[0].fill_between(episodes_train, training_rewards - training_ci, training_rewards + training_ci, color='blue', alpha=0.2)
axs[0].set_title('Training Phase')
axs[0].legend()
axs[0].set_ylabel('Reward')
axs[0].set_xlabel('Episodes')
print(episodes_test[0])
print(test_mean)
# Plot per la fase di test
axs[1].errorbar(0.4, test_mean, yerr=[[(test_mean - test_ci)], [(test_mean + test_ci)]], fmt='o', capsize=5, label='Test Reward', color='green')
axs[1].errorbar(0.6, baseline_mean, yerr=[[(baseline_std - baseline_ci)], [(baseline_mean + baseline_ci)]], fmt='o', capsize =5, label='Baseline Reward', color='red')
axs[1].set_xticks([])
axs[1].set_xlim(0, 1)
axs[1].set_title('Test Phase')
axs[1].legend()
axs[1].set_ylabel('Reward')
fig.suptitle("Agent 1")
plt.tight_layout()
plt.show()
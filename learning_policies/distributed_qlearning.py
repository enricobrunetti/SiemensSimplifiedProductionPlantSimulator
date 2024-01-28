import numpy as np

class QLearningAgent:
    def __init__(self, actions, n_products, alpha=0.1, gamma=0.9):
        self.actions = actions
        # learning rate
        self.alpha = alpha
        # discount factor
        self.gamma = gamma
        self.default_q_value = -n_products / (1 - self.gamma)
        self.q_table = {}

    def update_q_value(self, state, action, reward, next_state):
        current_q_value = self.q_table.get((tuple(state), action), self.default_q_value)
        next_max_q_value = max([self.q_table.get((tuple(next_state), a), self.default_q_value) for a in self.actions])
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_max_q_value - current_q_value)
        self.q_table[(tuple(state), action)] = new_q_value

    def select_action(self, state, exploration_prob, mask):
        temp_actions = [action for action, mask in zip(self.actions, mask) if mask != 0]
        if np.random.rand() < exploration_prob:
            print('random action choosen')
            return np.random.choice(temp_actions)
        else:
            q_values = [(self.q_table.get((tuple(state), a), self.default_q_value), a) for a in temp_actions]
            print(f'q-values: {q_values}')
            return max(q_values, key=lambda x: x[0])[1]
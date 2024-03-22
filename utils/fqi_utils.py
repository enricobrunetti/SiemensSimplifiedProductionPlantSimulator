from gymnasium import spaces

class MDP():
    def __init__(self, state_dim, actions):
        self.action_space = spaces.Discrete(len(actions))
        self.state_dim = state_dim
        self.action_dim = 1
        self.gamma = 0.99
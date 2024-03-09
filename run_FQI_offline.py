import json
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.trajectories_management import split_data_single_agent
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble import ExtraTreesRegressor

INPUT_DIR = "output/export_trajectories6_POSTPROCESSED.json"
CONFIG_PATH = "config/FQI_config.json"

# TO-DO: move this inside environment class
class MDP():
    def __init__(self):
        self.action_space = None
        self.state_dim = 105 
        self.action_dim = 1
        self.gamma = 0.99

if __name__ == '__main__':
    with open(CONFIG_PATH) as config_file:
        config = json.load(config_file)

    mdp = MDP()

    n_agents = config['n_agents']
    actions = config['available_actions']

    regressor_params = config['regressor_params']
    
    max_iterations = config['max_iterations']
    batch_size = config['batch_size']
    n_runs = config['n_runs']
    n_jobs = config['n_jobs']
    seed = config['seed']
    fit_params = config['fit_params']

    # FQI
    # TO-DO: before FQI fix absorbing (see trajectories_management)
    pi = []
    agents = []
    for i in range(n_agents):

        pi[i] = EpsilonGreedy(actions, ZeroQ(), 0)

        agents[i] = FQI(mdp, pi[i], verbose = True, actions = actions,
                        batch_size = batch_size, max_iterations = max_iterations,
                        regressor_type = ExtraTreesRegressor, **regressor_params)
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        agents[i].reset()
        _, _, _, r, s_prime, absorbing, sa, _ = split_data_single_agent(INPUT_DIR, i)

        for _ in range(max_iterations):
            agents[i]._iter(sa, r, s_prime, absorbing, **fit_params)

    example_state = sa[10][:-1]
    print(f'example_state: {example_state}')
    chosen_action = agents[0]._policy.sample_action(example_state)
    print('action {} has been chosen for state {}'.format(chosen_action, example_state))


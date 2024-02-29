import json
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.trajectories_management import split_data_single_agent
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble import ExtraTreesRegressor

INPUT_DIR = "output/export_trajectories6_POSTPROCESSED.json"

# TO-DO: move this inside environment class
class MDP():
    def __init__(self):
        self.action_space = None
        self.state_dim = 105 
        self.action_dim = 1
        self.gamma = 0.99

if __name__ == '__main__':

    mdp = MDP()

    actions = [6, 7, 8, 9, 10]

    regressor_params = {'n_estimators': 100,
                        'min_samples_split': 10}
    
    max_iterations = 1
    batch_size = 1
    n_runs = 20
    n_jobs = 10
    seed = None
    fit_params = {}

    # FQI
    # TO-DO: before FQI fix absorbing (see trajectories_management)
    pi = EpsilonGreedy(actions, ZeroQ(), 0)

    algorithm = FQI(mdp, pi, verbose = True, actions = actions,
                    batch_size = batch_size, max_iterations = max_iterations,
                    regressor_type = ExtraTreesRegressor, **regressor_params)
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    algorithm.reset()
    _, _, _, r, s_prime, absorbing, sa, _ = split_data_single_agent(INPUT_DIR, 0)

    for _ in range(max_iterations):
        algorithm._iter(sa, r, s_prime, absorbing, **fit_params)

    example_state = sa[10][:-1]
    print(f'example_state: {example_state}')
    chosen_action = algorithm._policy.sample_action(example_state)
    print('action {} has been chosen for state {}'.format(chosen_action, example_state))


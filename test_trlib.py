import json
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment
from utils.trajectories_management import extract_agent_trajectories, set_agents_state_observability, split_data_single_agent
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi import FQI
from sklearn.ensemble import ExtraTreesRegressor

#extract_agent_trajectories("prova", 5, 4)

#set_agents_state_observability("prova", 5, 4, 1)

class MDP():
    def __init__(self):
        self.action_space = None
        # ATTENTION: UNDERSTAND WHAT TO PLACE HERE
        self.state_dim = 118
        self.action_dim = 1
        self.gamma = 0.999

if __name__ == '__main__':

    mdp = MDP()

    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    regressor_params = {'n_estimators': 100,
                        'criterion': 'squared_error',
                        'min_samples_split':10}
    
    max_iterations = 1
    batch_size = 1
    n_runs = 20
    n_jobs = 10
    seed = None
    fit_params = {}

    # FQI
    pi = EpsilonGreedy(actions, ZeroQ(), 0)

    algorithm = FQI(mdp, pi, verbose = True, actions = actions,
                    batch_size = batch_size, max_iterations = max_iterations,
                    regressor_type = ExtraTreesRegressor, **regressor_params)
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    algorithm.reset()
    _, _, _, r, s_prime, absorbing, sa = split_data_single_agent("prova", 0)

    for _ in range(max_iterations):
        algorithm._iter(sa, r, s_prime, absorbing, **fit_params)

    example_state = sa[0][:-1]
    print(f'example_state: {example_state}')
    chosen_action = algorithm._policy.sample_action(example_state)
    print('action {} has been chosen for state {}'.format(chosen_action, example_state))



    '''t, s, a, r, s_prime, absorbing, sa = split_data_single_agent("prova", 0)
    print(f"t: {t}")
    print(len(t))
    print(f"s: {s}")
    print(len(s))
    print(f"s': {s_prime}")
    print(len(s_prime))
    print(f"a: {a}")
    print(len(a))
    print(f"r: {r}")
    print(len(r))
    print(f"sa: {sa}")
    print(len(sa))
    print(f"absorbing: {absorbing}")
    print(len(absorbing))'''

import numpy as np
from utils.graphs_utils import DistQAndLPIPlotter, FQIPlotter

model_path_runs = {}
model_path_runs[0] = 'models/20_units_increased_difficulty_1_product/reward5/True_True_25_True_500_False_IAP500/FQI/FQI_10.0_0_30_3_1_1_0.2ProvaMultipleEpsilon/run0'
model_path_runs[1] = 'models/20_units_increased_difficulty_1_product/reward5/True_True_25_True_500_False_IAP500/FQI/FQI_10.0_0_30_3_1_1_0.2ProvaMultipleEpsilon/run1'
model_path_runs[2] = 'models/20_units_increased_difficulty_1_product/reward5/True_True_25_True_500_False_IAP500/FQI/FQI_10.0_0_30_3_1_1_0.2ProvaMultipleEpsilon/run2'
model_path_runs[3] = 'models/20_units_increased_difficulty_1_product/reward5/True_True_25_True_500_False_IAP500/FQI/FQI_10.0_0_30_3_1_1_0.2ProvaMultipleEpsilon/run3'
model_path_runs[4] = 'models/20_units_increased_difficulty_1_product/reward5/True_True_25_True_500_False_IAP500/FQI/FQI_10.0_0_30_3_1_1_0.2ProvaMultipleEpsilon/run4'
multiple_exploration_probabilities = True
exploration_probabilities = [0.01, 0.1, 0.2, 0.5]
episodes_for_each_explor_for_iteration = 5
test_episodes_for_fqi_iteration = 21

plotter = FQIPlotter(model_path_runs, test_episodes_for_fqi_iteration, multiple_exploration_probabilities, exploration_probabilities, episodes_for_each_explor_for_iteration)
plotter.plot_performance_graph_multiple_epsilon()
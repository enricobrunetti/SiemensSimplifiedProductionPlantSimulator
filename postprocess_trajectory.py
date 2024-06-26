import json
from utils.trajectories_management import TrajectoryManager

INPUT_DIR = 'output/20_units/increased_difficulty_5_product_diff_config_true_true_50_true_500_false_500_no_action_mask/trajectory/run0'
OUTPUT_DIR = 'output/20_units/increased_difficulty_5_product_diff_config_true_true_50_true_500_false_500_no_action_mask/trajectory/POSTPROCESSEDObs2LightNeighborState.json'
CONFIG_PATH = "config/simulator_config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

tm = TrajectoryManager(INPUT_DIR, OUTPUT_DIR, config)
# manage reward type in config
#tm.compute_reward()
tm.remove_production_skill_trajectories()
tm.remove_action_masks()
# manage observability grade in the config
#tm.extract_agent_state_and_product_skills_for_DISTQ()
# for LPI mantain max observability, so we can extract k and beta neighbourhood for each step
tm.set_states_observability(restricted_neighbours_state=True)
tm.extract_agent_trajectories()
tm.save_trajectory()


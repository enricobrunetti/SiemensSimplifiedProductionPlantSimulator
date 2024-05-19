import json
from utils.trajectories_management import TrajectoryManager

INPUT_DIR = 'output/9_units_increased_difficulty_5_products_diff_config_random_test_for_FQI_semiMDP_standard_till_product_end_pos_shaping50/trajectory/run0'
OUTPUT_DIR = 'output/9_units_increased_difficulty_5_products_diff_config_random_test_for_FQI_semiMDP_standard_till_product_end_pos_shaping50/trajectory/POSTPROCESSED.json'
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
tm.set_states_observability()
tm.extract_agent_trajectories()
tm.save_trajectory()


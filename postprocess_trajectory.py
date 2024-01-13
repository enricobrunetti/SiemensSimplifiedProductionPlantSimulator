import json
from utils.trajectories_management import TrajectoryManager

INPUT_DIR = 'output/export_trajectories5.json'
OUTPUT_DIR = 'output/export_trajectories5_POSTPROCESSED.json'
CONFIG_PATH = "config/config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

tm = TrajectoryManager(INPUT_DIR, OUTPUT_DIR, config)
# manage reward type in config
tm.compute_reward()
tm.remove_production_skill_trajectories()
tm.remove_action_masks()
# manage observability grade in the config
tm.set_states_observability()
tm.extract_agent_trajectories()
tm.save_trajectory()


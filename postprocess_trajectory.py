import json
from utils.trajectories_management import TrajectoryManager

INPUT_DIR = 'output/export_trajectories2.json'
CONFIG_PATH = "config/config.json"

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

tm = TrajectoryManager(INPUT_DIR, config)
tm.remove_production_skill_trajectories()
tm.remove_action_masks()
tm.set_states_observability(1)
tm.extract_agent_trajectories()
tm.save_trajectory()


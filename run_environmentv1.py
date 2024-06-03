from production_plant_environment.env.production_plant_environment_v1 import ProductionPlantEnvironment
import json
import os

CONFIG_PATH = "config/simulator_config_env_v1_test.json"
with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)

model_path = "models/env_v1_test"
if not os.path.exists(model_path):
    os.makedirs(model_path)

env = ProductionPlantEnvironment(config, 0, model_path)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # invalid action masking is optional and environment-dependent
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None
        action = env.action_space(agent).sample(mask) # this is where you would insert your policy
    env.step(action)
env.close()
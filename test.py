import gymnasium
import numpy as np
from production_plant_environment.env.production_plant_environment_v0 import ProductionPlantEnvironment

with open('output/output.txt', 'w') as file:
    file.write('')

env = ProductionPlantEnvironment()

state = env.reset()

num_max_steps = 500

for step in range(num_max_steps):
    with open('output/output.txt', 'a') as file:
        file.write(f"***************Step{step}***************\n")
        file.write(f"Time: {state['time']}, Current agent: {state['current_agent']}\n")
        file.write(f"Agents busy: {state['agents_busy']}\n")
        file.write(f"Agents state: \n{state['agents_state']}\n")
        file.write(f"Products state: \n{state['products_state']}\n")
        file.write(f"Action mask: \n{state['action_mask']}\n")

    if np.all(state['action_mask'][state['current_agent']] == 0) or state['agents_busy'][state['current_agent']][0] == 1:
        # if no actions available -> do nothing
        action = 9
    else:
        actions = np.array(env.action_space)
        actions = actions[np.array(state['action_mask'][state['current_agent']]) == 1]
        action = np.random.choice(actions)

    state, reward, done, _ = env.step(action)

    with open('output/output.txt', 'a') as file:
        file.write(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}\n\n")

    if done:
        print("The episode is finished.")
        break
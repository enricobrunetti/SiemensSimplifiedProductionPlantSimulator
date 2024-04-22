# Simplified Siemens Simulator
## Introduction
The main purpose of this repo is to have a faster and lighter simulator than the one provided by Siemens in order to speed up learning processes.
## How to run
### Training
If you need to run a simulation in order to train a model the first thing to do is to open the ``simulator_config.json`` file and properly set the simulation parameters. Here a quick overview on them:
* ``algorithm``: this parameter refers to the algorithm used for the training. Avaiable options are **random**, **DistQ**, **LPI** and **FQI** (soon full working).
* ``test_model``: this is a boolean parameter which if is setted on **false** means that we are running a training session.
* ``test_model_name``: this parameter is not meaningful for training, you can leave it as it is.
* ``baseline_path``: this parameter has to be setted with the baseline path for the model that we wanto to traing, e.g. **"models/20_units/random_baseline"**.
* ``test_model_n_episodes``: not relevant for training.
* ``export_trajectories``: this parameter can be setted as **true** or **false**. If it is **true** the trajectories of the training will be saved in a **json** file. **NOTE**: this will slow down the training process a lot. Furthermore if you set it as **true** you will need to specify a path for the trajectories in ``trajectory_path``.
* ``output_log``: this parameter can be setted as **true** or **false**. If it is **true** the output of the training (representation of the state for each step) will be saved in a **txt** file. **NOTE:** this will slow down the training process a lot. Furthermore if you set it as **true** you will need to specify a path for the output files (will be one file for each episode) in ``output_path``.
* ``n_agents``: this parameter specify the number of agents (units) of the structure used for the simulation.
* ``n_products``:  this parameter specify the number of products of a bach (>1 if multiproduct).
* ``n_episodes``: specifies the number of training episodes (an episode is terminated when all the **n_products** have been produced).
* ``n_production_skills``: number of **production skills**.
* ``actions``: list containing alla the actions. You need to consider first all the **production skills** then **transport and defer skills** and in the end one more action for **skip turn action**, used if there is nothing to do with an agent in a time instant. E.g. if whe have ``"actions": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`` and ``"n_production_skills": 10`` it means that actions *from 0 to 9 are production skills, 10 is move up, 11 move right, 12 move down, 13 move left, 14 defer and 15 skip turn action*.
* ``available_actions``: list the actions choosable by the algorithm in the following order: *move up, move right, move down, move left, defer*. According to the example for ``actions`` it will be ``"available_actions": [10, 11, 12, 13, 14]``.
* ``action_time``: is a list that specify the default time instants required for the execution of each action in the same order as specified in ``actions``.
* ``action_energy``: same as above but considering the energy required for the execution of each action.
* ``custom_reward``: allow to specify the type of custom reward that you want to calculate and use in the simulation. Actually there are 5 types of custom rewards defined:
    1. ``reward1``: it gives to each availavle action (transport ones and defer) a reward value that corresponds to *-1 times* the number of products passed from the agent who played the action but not yet completed.
    2. ``reward2``: it gives to each availavle action (transport ones and defer) as reward value $100*\verb|n_times|*\gamma^t$, where $\verb|n_times|$ is the number of times that the agent who played the action seen the product and $t$ is the time instant on which the action is played.
    3. ``reward3``: it gives to each availavle action (transport ones and defer) as reward value of $-1 * (\verb|n_products| - \verb|n_not_completed_products|)$
    4. ``reward4``: equal to ``reward1`` but gives as reward value $-100$ until all the products of the banch have been supplied (are circulating). 
    5. ``reward5``: it is the **semi MDP** reward. It gives to each availavle action (transport ones and defer) a reward value that corresponds to the time instants required for the agent that played the action to have again the product. If the agent won't have again the product for the whole episode it gives as reward the time instants required to reach the end of the episode. **NOTE:** this reward con be used only if the **update_values** of **DistQ** or **LPI** is setted to be episodic and not at eache step. 
* ``alpha`` and ``beta``: are meaningful only if no custom reward is selected. This means that we are using default reward composed by a weighted mean between time required for the action and energy required for the action, where *alpha* is the weight of the time and *beta* is the weight of the energy.
* ``agents_connections``: it is a dictionary that associates to each agent the connections that it has in the 4 possible directions in the following order: up, right, down and left. Possible values are **null** if no other agent connected in that direction or the number of the agent connected.
* ``agents_skills``: it is a dictionary that associates to each agent a list of the production skills that can be performed by that specifice agent. No need to specify transport and defer actions since are assumed to be available by default.
* ``agents_skills_custom_duration``: it is a dictionary that associates to some agents another dictionary in which the key is a production skill and the value is the time required to complete the skill. It has to be specified only if the time to complete for that skill by that agent differs from the default time to complete of that skill specified in ``action_time``.
* ``supply_agent``: specifies the number of the agent that has the **supply** skill.
* ``supply_action``: specifies the number of the **supply** skill.
* ``agents_starting_state``: defines the state of each agent. It is a list that contains a list for each agent in which is contained a number of zeroes equal to the number of products. In this way the state will change and there will be a 1 in correspondence for a product and an agent when the agent has that product. To understand how it is work you can look at some output logs.
* ``products_starting_state``: each inner list corresponds to a production skill, in increasing order from 0. You have to put increasing numbers starting from one that represent the order of production skill that must be followed to produce the product.
* ``num_max_steps``: maximum number of steps before stopping the episode (assumed to be looped).
* ``observability_grade``: not relevant for the moment.
* ``gamma``: discount factor.

Then, in order to launch the training it is needed to execute the ``run_environment.py`` file.

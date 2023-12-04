from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector

import numpy as np

#TO-DO: move these matrices in a config file
AGENTS_STATUS = [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]

AGENTS_SKILLS = [[1, 0, 0, 0],
                 [0, 3, 0, 2],
                 [0, 0, 4, 2],
                 [0, 5, 0, 2],
                 [0, 0, 6, 2]]

PRODUCTS_STATUS =[[1, 2, 3, 4],
                  [1, 0, 2, 3],
                  [1, 3, 2, 4],
                  [1, 0, 0, 2],
                  [1, 3, 2, 4]]

AGENTS_CONNECTIONS = {0: [1, 2],
                      1: [0, 2, 4],
                      2: [0, 1, 3],
                      3: [2, 4],
                      4: [1, 3]}

class ProductionPlantEnvironment(AECEnv):
    
    metadata = {"render_modes": ["human"], "name": "production_plant_environment"}

    def __init__(self, render_mode=None):
        #TO-DO: make the number of agents a parameter to be assigned in a config file
        self.all_agents = ["agent_" + str(r) for r in range(5)]

        self.agent_name_mapping = dict(
            zip(self.all_agents, list(range(len(self.all_agents))))
        )

        #TO-CHECK: number of skills of power plant (in general) + number of possible trasnfers (ideally 4 directions) + defer action
        self._action_spaces = {agent: Discrete(9) for agent in self.all_agents}

        #TO-CHECK: owned product id and skill required
        self._observation_spaces = {
            agent: MultiDiscrete([5] * 2) for agent in self.all_agents
        }
        self.render_mode = render_mode
    
    def observation_space(self, agent):
        return MultiDiscrete([5] * 2)
    
    def action_space(self, agent):
        return Discrete(9)
    
    def render(self):
        #TO-DO
        pass
    def observe(self, agent):
        return np.array(self.observations[agent])
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):

        

        self.agents = self.all_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        
        #agent_selector allows cyclic stepping through the agents list
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        
        #CAPIRE QUESTO IF 
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = REWARD_MAP[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        

'''from pettingzoo import ParallelEnv

class ProductionPlantEnvironment(ParallelEnv):

    metadata = {
        "name": "power_plant_environment_v0",
    }

    def __init__(self, agents, agents_status, agents_skills, products_status):
        self.all_agents = agents
        self.agents_skills = agents_skills
        self.initial_agents_status = agents_status
        self.initial_products_status = products_status

        self.agents = None
        self.agents_status = None
        self.products_status = None
        self.timestep = None

    def reset(self, seed = None, options = None):
        self.agents_status = self.initial_agents_status.copy()
        self.products_status = self.initial_products_status.copy()
        self.timestep = 0
        self.agents = self.all_agents.copy()

        observations = {
            {
                'agents_skills': self.agents_skills,
                'agents_status': self.agents_status,
                'products_status': self.products_status
            }
        for a in self.agents
        }

        #TO-DO
        infos = {a: {} for a in self.agents}

        return observations, infos
        '''
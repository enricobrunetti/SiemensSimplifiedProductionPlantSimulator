import numpy as np
import random
import json

class AgentManager:
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent):
        self.agents[agent.name] = agent

    def get_q_values_of_agent(self, agent_name):
        if agent_name in self.agents:
            return self.agents[agent_name].q_values
        else:
            return None

    def update_q_values_of_agent(self, agent_name, new_q_values):
        if agent_name in self.agents:
            self.agents[agent_name].q_values = new_q_values

    def get_updated_q_values(self, requesting_agent_name, target_agent_name):
        if target_agent_name in self.agents:
            return self.get_q_values_of_agent(target_agent_name)
        else:
            return None


class Agent:
    def __init__(self, name, q_values, agent_manager):
        self.name = name
        self.q_values = q_values
        self.agent_manager = agent_manager

    def request_updated_q_values(self, target_agent_name):
        return self.agent_manager.get_updated_q_values(self.name, target_agent_name)
    
    def update_q_values(self, new_values):
        self.q_values = new_values


# Creazione del gestore degli agenti
agent_manager = AgentManager()

# Creazione e registrazione degli agenti
agent1 = Agent("Agent1", {'action1': 0.5, 'action2': 0.3}, agent_manager)
agent2 = Agent("Agent2", {'action1': 0.7, 'action2': 0.2}, agent_manager)

agent_manager.register_agent(agent1)
agent_manager.register_agent(agent2)

# Richiesta dei q_values più aggiornati di agent2 da parte di agent1
agent2_updated_q_values = agent1.request_updated_q_values("Agent2")
print("Q_values of Agent2 requested by Agent1:", agent2_updated_q_values)

agent2.update_q_values({'action1': 10, 'action2': 2})

agent2_updated_q_values = agent1.request_updated_q_values("Agent2")
print("Q_values of Agent2 requested by Agent1:", agent2_updated_q_values)





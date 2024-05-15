def get_next_agent_number(agent_num, action, actions, agents_connections):
    action -= actions[0]
    if action == 4:
        return agent_num
    return agents_connections[agent_num][action] 
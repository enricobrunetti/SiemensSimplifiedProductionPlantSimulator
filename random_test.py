import numpy as np
import random
                  
agents_connections = {0: [None, 1, 2, None],
                    1: [None, 4, 2, 0],
                    2: [1, 3, None, 0],
                    3: [4, None, None, 2],
                    4: [None, None, 3, 1]}

agents_state = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [1, 0, 0, 0]])

action_mask = {}
for i in range(5):
    action_mask[i] = np.zeros(10)


#print(all(elem == 0 for elem in agents_state[agents_connections[1][3]]) and agents_connections[1][3] != None)

'''for agent in agents_connections:
    for i in range(4, 7):
        if agents_connections[agent][i - 4] != None and not all(elem == 0 for elem in agents_state[agents_connections[agent][i - 4]]):
            action_mask[agent][i] = 0
        elif agents_connections[agent][i - 4] != None and all(elem == 0 for elem in agents_state[agents_connections[agent][i - 4]]) and not all(elem == 0 for elem in agents_state[agent]):
            action_mask[agent][i] = 1'''




def hasProduct(agent):
    return not np.all(agents_state[agent] == 0)

print(agents_connections[1][3] != None and not hasProduct(0))
print(not hasProduct(0))

for agent in agents_connections:
    if hasProduct(agent):
        for i in range(4, 8):
            print(agents_connections[agent][i-4] != None and (not hasProduct(agents_connections[agent][i - 4])))
            if agents_connections[agent][i-4] != None and hasProduct(agents_connections[agent][i - 4]):
                action_mask[agent][i] = 0
            elif agents_connections[agent][i-4] != None and (not hasProduct(agents_connections[agent][i - 4])):
                action_mask[agent][i] = 1

for action in action_mask:
    print(f"{action}: {action_mask[action]}")


print([i for i in range(4, 7)])

print(action_mask[2][:6])
print(np.all(action_mask[1][4:] == 0))
import random
import main
import pprint


grid = [['.' for x in range(main.MAX_WIDTH)] for y in range(main.MAX_HEIGHT)]

# Define the state space
states = []
for x in range(main.MAX_WIDTH):
    for y in range(main.MAX_HEIGHT):
        states.append((x, y))

# Define the positions and radii of the green and red circles
green_circle_pos = (150, 150)
green_circle_radius = 50
red_circle_pos = (500, 400)
red_circle_radius = 50

# Set the Q-values for reaching the green circle
for x in range(main.MAX_WIDTH):
    for y in range(main.MAX_HEIGHT):
        if (x - green_circle_pos[0])**2 + (y - green_circle_pos[1])**2 <= green_circle_radius**2:
            main.Q[x, y, :] = 10

# Set the Q-values for reaching the red circle
for x in range(main.MAX_WIDTH):
    for y in range(main.MAX_HEIGHT):
        if (x - red_circle_pos[0])**2 + (y - red_circle_pos[1])**2 <= red_circle_radius**2:
            main.Q[x, y, :] = -10


for x in range(100, 200):
    for y in range(100, 200):
        if (x - 150)**2 + (y - 150)**2 <= 50**2:
            grid[x][y] = 'G'
for x in range(450, 550):
    for y in range(350, 450):
        if (x - 500)**2 + (y - 400)**2 <= 50**2:
            grid[x][y] = 'R'

def select_action(robot, epsilon):
    # Select an action randomly with probability epsilon
    robot.position = (int(robot.position[0]), int(robot.position[1]))

    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Return a random action

    # Otherwise, select the action with the highest Q-value
    action_values = main.Q[robot.position[0]][robot.position[1]]
    return max(enumerate(action_values), key=lambda x: x[1])[0]


def Qlearning(robot, alpha=0.1, gamma=0.9):
    # Loop through multiple episodes
    # Set the initial state
    # Select an action based on the Q-values of the current state
    action = select_action(robot, 0.5)

    # Take the action and observe the reward and the next state
    reward, next_position = take_action(robot, action)

    # Update the Q-value of the current state and action
    main.Q[robot.position[0]][robot.position[1]][action] = main.Q[robot.position[0]][robot.position[1]][action] + alpha * (reward + gamma * max(main.Q[next_position[0]][next_position[1]]) - main.Q[robot.position[0]][robot.position[1]][action])
    # Set the current state to the next state
    robot.position = next_position 
    
def take_action(robot, action):
    # Update the current state based on the action
    if action == 0:  # Up
        next_state = (robot.position[0], robot.position[1] - 1)
    elif action == 1:  # Down
        next_state = (robot.position[0], robot.position[1] + 1)
    elif action == 2:  # Left
        next_state = (robot.position[0] - 1, robot.position[1])
    elif action == 3:  # Right
        next_state = (robot.position[0] + 1, robot.position[1])

    next_state = (int(next_state[0]), int(next_state[1]))

    # Check if the new state is outside the boundaries of the 2D space
    if next_state[0] < 0 or next_state[0] >= main.MAX_WIDTH or next_state[1] < 0 or next_state[1] >= main.MAX_HEIGHT:
        # Return a reward of -1 if the state is outside the boundaries
        return -1, robot.position

    # Check if the new state is a green or red circle
    if grid[next_state[0]][next_state[1]] == 'G':
        # Return a reward of +1 if the state is a green circle
        return 1, next_state
    elif grid[next_state[0]][next_state[1]] == 'R':
        # Return a reward of -1 if the state is a red circle
        return -1, next_state
    elif grid[next_state[0]][next_state[1]] == 'C':
        # Return a reward of -1 if the state is a red circle
        return 5, next_state

    # Return a reward of 0 if the state is neither a green nor a red circle
    return 0, next_state

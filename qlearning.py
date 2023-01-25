import random
import main
import torch
import numpy as np
import math

# Define the neural network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(18, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 4)
)
device = torch.device("cuda")

# Define the loss function and the optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

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
            for i in range(4):
                main.Q[x][y][i] = 10

# Set the Q-values for reaching the red circle
for x in range(main.MAX_WIDTH):
    for y in range(main.MAX_HEIGHT):
        if (x - red_circle_pos[0])**2 + (y - red_circle_pos[1])**2 <= red_circle_radius**2:
            for i in range(4):
                main.Q[x][y][i] = -10


for x in range(100, 200):
    for y in range(100, 200):
        if (x - 150)**2 + (y - 150)**2 <= 50**2:
            grid[x][y] = 'G'
for x in range(450, 550):
    for y in range(350, 450):
        if (x - 500)**2 + (y - 400)**2 <= 50**2:
            grid[x][y] = 'R'


def heuristic(position):
    x, y = position
    x_green, y_green = green_circle_pos
    return np.sqrt((x - x_green)**2 + (y - y_green)**2)


def select_action_EPSILON(robot, epsilon = 0.5):
    # Select an action randomly with probability epsilon
    robot.position = (int(robot.position[0]), int(robot.position[1]))

    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Return a random action

    # Otherwise, select the action with the highest Q-value
    action_values = main.Q[robot.position[0]][robot.position[1]]
    return max(enumerate(action_values), key=lambda x: x[1])[0]


def select_action(robot, epsilon, other_robots):
    # Convert the robot's position to a 1D array
    state = np.array([robot.position[0], robot.position[1]] + [r.position[0] for r in other_robots] + [r.position[1] for r in other_robots])
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state, dtype=torch.float)

    # Use the neural network to predict the Q-values for each action
    q_values = model(state)
    
    # Select the action with the highest Q-value, with probability (1-epsilon)
    best_action = torch.argmax(q_values).item()
    if random.random() > epsilon:
        return best_action

    # Otherwise, select a random action with probability epsilon
    return random.randint(0, 3)

def Qlearning(robots, alpha=0.1, gamma=0.9):
    for robot in robots:
        other_robots = [x for x in robots if x != robot]
        action = select_action(robot, 0.5, [x for x in robots if x != robot])
        action_tensor = torch.tensor(action, dtype=torch.long)

        # Take the action and observe the reward and the next state
        reward, next_position = take_action(robot, action)

        current_state = np.array([robot.position[0], robot.position[1]] + [r.position[0] for r in other_robots] + [r.position[1] for r in other_robots])
        current_state = np.expand_dims(current_state, axis=0)
        next_state = np.array([next_position[0], next_position[1]] + [r.position[0] for r in other_robots] + [r.position[1] for r in other_robots])
        next_state = np.expand_dims(next_state, axis=0)

        # Convert the current state and next state to 1D arrays
        current_state = np.array([robot.position[0], robot.position[1]] + [r.position[0] for r in other_robots] + [r.position[1] for r in other_robots])
        current_state = np.expand_dims(current_state, axis=0)
        next_state = np.array([next_position[0], next_position[1]] + [r.position[0] for r in other_robots] + [r.position[1] for r in other_robots])
        next_state = np.expand_dims(next_state, axis=0)

        # Use the neural network to predict the Q-values for the current and next states
        current_q_values = model(torch.tensor(current_state, dtype=torch.float))
        next_q_values = model(torch.tensor(next_state, dtype=torch.float))

        target_q_values = current_q_values.clone()
        target_q_values[0][action] = reward + gamma * next_q_values.max()

        # Compute the loss
        output = loss_fn(current_q_values, target_q_values)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Compute the gradients
        output.backward()
        
        # Update the model weights
        optimizer.step()
        
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
        return -1000, robot.position

    # Check if the new state is a green or red circle
    if grid[next_state[0]][next_state[1]] == 'G':
        # Return a reward of +1 if the state is a green circle
        print("HIT GREEN!")
        return 1000, next_state
    elif grid[next_state[0]][next_state[1]] == 'R':
        # Return a reward of -1 if the state is a red circle
        return -1000, next_state

    # Return a reward of 0 if the state is neither a green nor a red circle
    return -heuristic(next_state), next_state

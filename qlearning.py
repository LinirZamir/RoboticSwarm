import random
import main
import torch
import numpy as np
import math

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


def select_action(robots, robot, Q_table, epsilon = 0.5):
    # Select an action randomly with probability epsilon
    current_state = np.array([robot.position[0], robot.position[1]] + [r.position[0] for r in robots if r != robot] + [r.position[1] for r in robots if r != robot])
    current_state = np.expand_dims(current_state, axis=0)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Return a random action

    # Otherwise, select the action with the highest Q-value
    encoded = Q_table[current_state[0]][current_state[1]][4*robots.index(robot):4*(robots.index(robot)+1)]
    encoded = np.array(encoded)
    encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
    predicted = robot.model.predict(encoded_reshaped).flatten()
    action = np.argmax(predicted)

    return action


def Qlearning(robots, Q_table, epsilon=0.4, gamma=0.9):
    for robot in robots:
        current_state = np.array([robot.position[0], robot.position[1]] + [r.position[0] for r in robots if r != robot] + [r.position[1] for r in robots if r != robot])
        current_state = np.expand_dims(current_state, axis=0)
        current_state_discrete = discretize_state(current_state, num_bins=40)
        action = select_action(robots, robot, Q_table, epsilon) # use the neural network to select the action

        # Take the action and observe the reward and the next state
        reward, next_position = take_action(robot, action)
        robot.total_reward += reward

        next_state = np.array([next_position[0], next_position[1]] + [r.position[0] for r in robots if r != robot] + [r.position[1] for r in robots if r != robot])
        next_state = np.expand_dims(next_state, axis=0)
        next_state_discrete = discretize_state(next_state, num_bins=40)

        # Update the Q-value for the current state-action pair
        Q_table[current_state[0]][current_state[1]][4*robots.index(robot) + action] = \
        Q_table[current_state[0]][current_state[1]][4*robots.index(robot) + action] + robot.alpha * \
        (reward + gamma * np.max(Q_table[next_state[0]][next_state[1]][4*robots.index(robot):4*(robots.index(robot)+1)]) - \
        Q_table[current_state[0]][current_state[1]][4*robots.index(robot) + action])
        # Move on to the next state
        current_state = next_state


def take_action(robot, action):
    # Update the current state based on the action
    current_state = robot.position
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
        return -10, robot.position

    # Check if the new state is a green or red circle
    if grid[next_state[0]][next_state[1]] == 'G':
        # Return a reward of +1 if the state is a green circle
        print("HIT GREEN!")
        reward = 10
    elif grid[next_state[0]][next_state[1]] == 'R':
        # Return a reward of -1 if the state is a red circle
        reward = -10
    else:
        # Return a reward of 0 if the state is neither a green nor a red circle
        reward = -heuristic(next_state)

    return reward, next_state


def discretize_state(current_state, num_bins):
    # Create an array of bin edges for each feature in the state
    bin_edges = [np.linspace(np.min(current_state[:, i]), np.max(current_state[:, i]), num_bins + 1) for i in range(current_state.shape[1])]

    # Use np.digitize to find the indices of the bins that each feature falls into
    bin_indices = [np.digitize(current_state[:, i], bin_edges[i]) for i in range(current_state.shape[1])]

    # Use np.ravel_multi_index to convert the multi-dimensional indices into a single flat index
    flat_indices = np.ravel_multi_index(bin_indices, [num_bins]*current_state.shape[1])

    return flat_indices

def train(model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
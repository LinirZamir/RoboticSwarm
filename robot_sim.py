import numpy as np 
import random
from tensorflow import keras
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense


class Robot:
    def __init__(self, id, state, dimension,position_range):
        self.id = id
        self.session = None
        self.state = state
        self.pbest = float('inf')

        ## Positioning + PSO
        self.position = [random.uniform(position_range[0], position_range[1]) for j in range(dimension)]
        self.dimension = dimension
        self.position_range = position_range
        self.velocity = np.array([0.0 for i in range(self.dimension)])
        self.current_velocity = None
        self.total_reward = 0
        self.model = None

    def build_model(self, num_robots):
        model = Sequential()
        model.add(Dense(64, input_dim=2*(num_robots-1), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        self.model =  model

    def send_message(self, topic, message):
        # Send a message on the specified topic
        pass

    def update_state(self, newstate):
        self.state = newstate
        # Update the state of the robot based on the messages it has received and the actions it has taken
        pass

    def update_velocity(self, bpest, w_min=0.5, max=1.0, c=0.1):
        # Randomly generate r1, r2 and inertia weight from normal distribution
        r = random.uniform(0,max)
        w = random.uniform(w_min,max)
        c1 = c
        a = 0.05
        b = 0.45
        # Calculate new velocity
        for i in range(self.dimension):
            self.velocity[i] = w*self.velocity[i] +c1*r*(bpest[i]-self.position[i])

    def update_position(self):
        # Move particles by adding velocity
        self.position = self.position + self.velocity

    def reset_pos(self):
        self.position =  [random.uniform(self.position_range[0], self.position_range[1]) for j in range(self.dimension)]
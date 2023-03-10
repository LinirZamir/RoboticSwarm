import numpy as np 
import random
import math

class Robot:
    def __init__(self, id, state, dimension,position_range):
        self.id = id
        self.flower_locations = []
        self.session = None
        self.state = state
        self.pbest = float('inf')
        self.position_range = position_range
        
        ## Positioning + PSO
        self.position = [random.uniform(position_range[0], position_range[1]) for j in range(dimension)]
        self.dimension = dimension
        self.position_range = position_range
        self.velocity = np.array([0.0 for i in range(self.dimension)])
        self.current_velocity = None

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
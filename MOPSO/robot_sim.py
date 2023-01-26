import numpy as np 
import random
import math

class Robot:
    def __init__(self, id, state, camera, yolo, dimension,position_range):
        self.id = id
        self.camera = camera
        self.yolo = yolo
        self.flower_locations = []
        self.session = None
        self.state = state
        self.pbest = float('inf')
        
        ## Positioning + PSO
        self.position = [random.uniform(position_range[0], position_range[1]) for j in range(dimension)]
        self.dimension = dimension
        self.position_range = position_range
        self.velocity = np.array([0.0 for i in range(self.dimension)])
        self.current_velocity = None

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

    def update_random_position(self, max_width, max_height):
        # Create a list of directions
        directions = ["up", "down", "left", "right"]

        # Generate a random integer between 0 and 3
        rand_int = random.randint(0, 3)

        # Use the random integer to determine the direction of movement
        direction = directions[rand_int]

        if direction == "up":
        # Move the robot up
            if self.position[1]<abs(max_height):
                self.position[1]+=1
        elif direction == "down":
        # Move the robot down
            if self.position[1]<abs(max_height):
                self.position[1]-=1
        elif direction == "left":
        # Move the robot left
            if self.position[0]<abs(max_width):
                self.position[0]-=1
        else:
        # Move the robot right
            if self.position[0]<abs(max_width):
                self.position[0]+=1

    def update_position(self):
        # Move particles by adding velocity
        self.position = self.position + self.velocity

    def reset_pos(self):
        self.position =  [random.uniform(self.position_range[0], self.position_range[1]) for j in range(self.dimension)]
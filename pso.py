import numpy as np
import math
import random

def fitness_function(x,y):
    ## return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)
    f1=x+2*-y+3
    f2=2*x+y-8
    z = f1**2+f2**2
    return z


# Hyper-parameter of the algorithm


def update(robots, dimension, fitness_criterion):
    c1 = c2 = 0.1
    a = 0.05
    b = 0.45
    w = 0.8
    pc = []
    pbest_fitness = []
    "Function to do one iteration of particle swarm optimization"
    # Update params
    for p in robots:
        if fitness_function(p.position[0],p.position[1])<p.pbest:
            p.pbest = fitness_function(p.position[0],p.position[1])
        pbest_fitness.append(p.pbest)
    gbest_index = np.argmin(pbest_fitness)
    if np.average(pbest_fitness) <= fitness_criterion:
        print("Found")
        return gbest_index
    else:
        for rob in robots:
            pbest = []
            for i in range(dimension):
                pc = 0.5 #a + b * ((math.exp((10 * rob.id) / (len(robots) - 1)) - 1) / math.exp(10) - 1)
                pc_rand = random.random()
                if(pc_rand<pc):
                    num1 = random.randint(0, len(robots)-1)
                    num2 = random.randint(0, len(robots)-1)
                    if fitness_function(robots[num1].position[0],robots[num1].position[1])<fitness_function(robots[num2].position[0],robots[num2].position[1]):
                        pbest.append(robots[num1].position[i])
                    else:
                        pbest.append(robots[num2].position[i])
                else:
                    pbest.append(rob.position[i])
            rob.update_velocity(pbest,w)
            rob.update_position()
    return -1
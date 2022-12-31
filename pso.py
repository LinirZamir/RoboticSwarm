import numpy as np
import math
import random


def schwefel(inputs):
    return 418.9829* len(inputs) - sum(x * math.sin(math.sqrt(abs(x))) for x in inputs)

def griewank(inputs):
    n = len(inputs)
    sum_term = sum(x**2 for x in inputs)
    prod_term = math.prod(math.cos(x / math.sqrt(i+1)) for i, x in enumerate(inputs))
    return 1 + sum_term / 4000 - prod_term

def rosenbrock_function(inputs):
    total = 0
    for i in range(len(inputs)-1):
        total+= ( (1 - inputs[i])**2 + 100*(inputs[i+1] - inputs[i]**2)**2)
    return total

def sphere_problem(inputs):
    total = 0
    for element in inputs:
        total += element**2
    return (total)

def general_problem(inputs):
    f1=inputs[0]+2*-inputs[1]+3
    f2=2*inputs[0]+inputs[1]-8
    z = f1**2+f2**2
    return z


def fitness_function(inputs):
    return general_problem(inputs)


# Hyper-parameter of the algorithm


def CLPSO(robots, dimension, fitness_criterion):
    c1 = c2 = 0.1
    a = 0.05
    b = 0.45
    w = 0.8
    pc = []
    pbest_fitness = []
    "Function to do one iteration of particle swarm optimization"
    # Update params
    for p in robots:
        if fitness_function(p.position)<p.pbest:
            p.pbest = fitness_function(p.position)
        pbest_fitness.append(p.pbest)
    gbest_index = np.argmin(pbest_fitness)
    if np.average(pbest_fitness) <= fitness_criterion:
        print("Found")
        return gbest_index
    else:
        for rob in robots:
            pbest = []
            for i in range(dimension):
                pc = a + b * ((math.exp((10 * rob.id-1) / (len(robots) - 1)) - 1)) / (math.exp(10) - 1)
                pc_rand = random.random()
                if(pc_rand<pc):
                    num1 = random.randint(0, len(robots)-1)
                    num2 = random.randint(0, len(robots)-1)
                    if fitness_function(robots[num1].position)<fitness_function(robots[num2].position):
                        pbest.append(robots[num1].position[i])
                    else:
                        pbest.append(robots[num2].position[i])
                else:
                    pbest.append(rob.position[i])
            rob.update_velocity(pbest,w)
            rob.update_position()
    return -1
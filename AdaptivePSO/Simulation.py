import pygame
import numpy as np
from Blank_PSO import adaptive_pso, evaluate_fitness, update_velocity_and_position, compute_trust


# Initialize pygame
pygame.init()

# Set the size of the window
size = (700, 500)
screen = pygame.display.set_mode(size)

# Create an empty grid
grid = np.zeros((size[0], size[1]))

# Create obstacles
for i in range(num_obstacles):
    obstacle_x = np.random.randint(0, size[0])
    obstacle_y = np.random.randint(0, size[1])
    obstacle_size = np.random.randint(10, 50)
    pygame.draw.rect(screen, (255, 0, 0), (obstacle_x, obstacle_y, obstacle_size, obstacle_size))
    grid[obstacle_x:obstacle_x+obstacle_size, obstacle_y:obstacle_y+obstacle_size] = 1
# Create dynamic obstacles

for i in range(num_dynamic_obstacles):
    dynamic_obstacle_x = np.random.randint(0, size[0])
    dynamic_obstacle_y = np.random.randint(0, size[1])
    dynamic_obstacle_size = np.random.randint(10, 50)
    pygame.draw.circle(screen, (255, 0, 0), (dynamic_obstacle_x, dynamic_obstacle_y), dynamic_obstacle_size)
    grid[dynamic_obstacle_x-dynamic_obstacle_size:dynamic_obstacle_x+dynamic_obstacle_size, dynamic_obstacle_y-dynamic_obstacle_size:dynamic_obstacle_y+dynamic_obstacle_size] = 2

# Initialize the position and velocities of the particles
positions = ...
velocities = ...

# Run the adaptive PSO algorithm
results = adaptive_pso(positions, velocities, grid, evaluate_fitness, update_velocity_and_position, compute_trust, non_dominated_sorting)

# Update the position of the autonomous system on the screen
autonomous_system_x, autonomous_system_y = results[-1][0]
pygame.draw.circle(screen, (0, 255, 0), (autonomous_system_x, autonomous_system_y), 5)

# Update the position of dynamic obstacles
update_dynamic_obstacles(grid)


# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Run the adaptive PSO algorithm again
    results = adaptive_pso(positions, velocities, grid, evaluate_fitness, update_velocity_and_position, compute_trust, non_dominated_sorting)
    
    # Update the position of the autonomous system on the screen
    autonomous_system_x, autonomous_system_y = results[-1][0]
    pygame.draw.circle(screen, (0, 255, 0), (autonomous_system_x, autonomous_system_y), 5)
    pygame.display.update()

    # Update the position of dynamic obstacles
    update_dynamic_obstacles(grid)

    # Wait for some time before running the algorithm again
    pygame.time.wait(1000)
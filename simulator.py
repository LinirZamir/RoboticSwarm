import pygame
import math 
import random
from main import MAX_HEIGHT, MAX_WIDTH

class Robot:
    def __init__(self, id, x, y, size, color):
        self.id = id
        self.x = x
        self.y = y
        self.size = size
        self.color = color

    def draw(self, screen):
        # Use Pygame's draw.circle function to draw a circle on the screen
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

class DynamicObstacle:
    def __init__(self, x, y, size, color, speed, direction):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed
        self.direction = direction 

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

    def update(self, window_size):
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)
        if self.x < self.size or self.x > window_size[0] - self.size:
            self.direction = math.pi - self.direction + + random.uniform(0, 2*math.pi)
        if self.y < self.size or self.y > window_size[1] - self.size:
            self.direction = - self.direction + random.uniform(0, 2*math.pi)



def simulator(robot_list):

    local_bots = []
    # Define the obstacles
    obstacles = []
    obstacles.append((100, 100, 50))
    obstacles.append((500, 400, 50))

    # Create a dynamic obstacle and add it to the list of obstacles
    dynamic_obstacle = DynamicObstacle(200, 200, 50, (255, 0, 0), 0.02, math.pi/2)
    obstacles = [dynamic_obstacle]


    # Define a function to draw the robots on the screen
    def draw_robots(screen,normalize_x,normalize_y):
        for bot in robot_list:
            found = False
            for local_bot in local_bots:
                if bot.id == local_bot.id:
                    local_bot.x = bot.position[0]+normalize_x
                    local_bot.y = bot.position[1]+normalize_y
                    found = True
            if not found:
                local_bots.append(Robot(bot.id, bot.position[0], bot.position[1], 5, (0,0,0)))

        for robot in local_bots:
            robot.draw(screen)


    # Initialize Pygame
    pygame.init()
    # Set the window size
    window_size = (MAX_WIDTH, MAX_HEIGHT)

    # Create the window
    screen = pygame.display.set_mode(window_size)

    # Set the window title
    pygame.display.set_caption('Robotic Swarm')

    # Define the center of the screen
    normalize_x = 0 # window_size[0] / 2
    normalize_y = 0 # window_size[1] / 2

    # Run the game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill((255, 255, 255))

        # Update the position of the dynamic obstacle
        dynamic_obstacle.update(window_size)

        # Draw the obstacles
        for obstacle in obstacles:
            obstacle.draw(screen)
            
        # Draw the robots
        draw_robots(screen,normalize_x,normalize_y)

        # Update the screen
        pygame.display.flip()

    # Shut down Pygame
    pygame.quit()
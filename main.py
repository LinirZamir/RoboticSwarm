import threading
from robot_sim import Robot
import time 
import signal
import simulator
import numpy as np



global_simulator_thread = None
global_keypress = 0

DIMENSION = 2
MAX_WIDTH = 640
MAX_HEIGHT = 640
NUM_EPISODES = 100

NUM_BOTS = 10
SIMULATION_TIME = 50

stop_flag = threading.Event()


def main():
    import qlearning



    global global_simulator_thread
    robot_list = []
    Q_table = np.zeros((640, 640, 2*NUM_BOTS)) # create a Q table with the correct dimensions


    for i in range(1,NUM_BOTS):
        new_rob = Robot(i, np.zeros((MAX_WIDTH, MAX_HEIGHT)),DIMENSION,[MAX_WIDTH/2-100,MAX_HEIGHT/2+100])
        new_rob.build_model(NUM_BOTS)
        robot_list.append(new_rob)

    # Start the simulator thread
    global_simulator_thread = threading.Thread(target=simulator.simulator, args=(robot_list,))
    global_simulator_thread.start()

    # Perform Q-learning for a set number of episodes
    for episode in range(NUM_EPISODES):
        for bot in robot_list:
            bot.total_reward = 0
            bot.reset_pos()
        start_time = time.time()
        while time.time() - start_time < SIMULATION_TIME:
            qlearning.Qlearning(robot_list, Q_table)


def handler(signum, frame):
    global global_simulator_thread

    print("shutting down...")
    stop_flag.set()
    global_simulator_thread.join()
    print("shutting down successful")
    exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    main()


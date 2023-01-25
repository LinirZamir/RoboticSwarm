import paho.mqtt.client as mqtt
import threading
from robot_sim import Robot
import time 
import signal
import simulator
from pso import CLPSO
import msvcrt
import random


global_simulator_thread = None
global_keypress = 0

DIMENSION = 2
MAX_WIDTH = 640
MAX_HEIGHT = 640

robot_list = []
stop_flag = threading.Event()


Q = [[[random.uniform(-1, 1) for action in range(4)] for x in range(MAX_WIDTH)] for y in range(MAX_HEIGHT)]


def main():
    import qlearning
    global global_simulator_thread

    for n in range(1,10):
        robot_list.append(Robot(n,0,"camera","yolo",DIMENSION,[MAX_WIDTH/2-100,MAX_HEIGHT/2+100]))

    print("connecting")
    global_simulator_thread = threading.Thread(target=simulator.simulator,args=(robot_list,))
    global_simulator_thread.start()

    counter = 0
    found = 0
    #while found == 0:
     #           time.sleep(0.01)
      #          for bot in robot_list:
       #             bot.update_random_position(MAX_WIDTH, MAX_HEIGHT)"""
    for i in range(100):
        for bot in robot_list:
            bot.reset_pos()
        start_time = time.time()
        while time.time() - start_time < 60:
            qlearning.Qlearning(robot_list)
                # sol = CLPSO(robot_list,DIMENSION,10e-3)
                #counter=counter+1
                #if(sol != -1):
                #    print(f"Reached Maxima Minima! {robot_list[sol].position}; Total iterations: {counter}")
                #    found = 1
            #time.sleep(0.004)

def handler(signum, frame):
    global global_simulator_thread

    print("shutting down...")
    stop_flag.set()
    global_simulator_thread.join()
    print("shutting down successful")
    exit(1)

# Define the function to be triggered by the keyboard input

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    main()

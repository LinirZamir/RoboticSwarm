import paho.mqtt.client as mqtt
import threading
from robot_sim import Robot
import time 
import signal
import simulator
from pso import update


global_simulator_thread = None

DIMENSION = 2

robot_list = []
stop_flag = threading.Event()

def main():
    global global_simulator_thread

    for n in range(1,100):
        robot_list.append(Robot(n,0,"camera","yolo",DIMENSION,[-100,100]))

    print("connecting")
    global_simulator_thread = threading.Thread(target=simulator.simulator,args=(robot_list,))
    global_simulator_thread.start()

    counter = 0
    while True:
        time.sleep(0.01)
        sol = update(robot_list,DIMENSION,10e-4)
        counter=counter+1
        if(sol != -1):
            print(f"Reached Maxima Minima! {robot_list[sol].position}; Total iterations: {counter}")
            break

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

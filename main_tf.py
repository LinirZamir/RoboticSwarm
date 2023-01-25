from ddpg import Agent
import gymnasium as gym 
import numpy as np
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    agent = Agent(alpha=0.0001, beta = 0.001, input_dims=[3], tau = 0.001, env = env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=1)

    score_history = []
    np.random.seed(0)
    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs,act,reward,new_state,int(done))
            agent.learn()
            score+= reward
            obs = new_state
        score_history.append(score)
        print(f'episode  {i} score {score}, 100 game average: {np.mean(score_history[-100:])}')

    filename = 'pendulum.png'
    plotLearning(score_history, filename, window=100)


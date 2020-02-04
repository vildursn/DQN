import Agent_cont as Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.models import load_model

model = load_model("Target_model_040220.h5")


env = gym.make('CartPole-v0')
action_space = env.action_space.n
agent = Agent.Pretrained_Agent(model, action_space)
obs_space = 4

obs = env.reset()
r=0
done = False
tot_r = []
for i in range(0,10):
    done = False
    obs = env.reset()
    r = 0
    while not done:
        #env.render()
        action = np.argmax(agent.get_q_values_target_model(obs))
        obs_next, reward, done, _ = env.step(action)
        r += reward
        obs=obs_next
    tot_r.append(r)

print(np.average(tot_r))

done = False
obs = env.reset()
while not done:
    env.render()
    action = np.argmax(agent.get_q_values_target_model(obs))
    obs_next, reward, done, _ = env.step(action)
    obs=obs_next

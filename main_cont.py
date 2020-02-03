import Agent_cont as Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# 100000 episoder, 1000 replay memory size, 100 batch size tar over ent ime....
NUM_EPISODES = 1000
LEARNING_RATE = 0.005
DISCOUNT = 0.9
EPSILON = 0.4
EPSILON_DECAY = 0.002
REPLAY_MEMORY_SIZE = 100
TRAIN_BATCH_SIZE = 32
UPDATE_TARGET_NUM = 10
HIDDEN_LAYERS_DIM = [100,100]
AVG_EVERY_NUM = 1


#USE THE TARGET MODEL FOR ACTION SELECTION, AND THE REGULAR MODEL FOR ACTION EVALUATION


def collect_experiences():
    while agent.replay_memory.is_replay_memory_full() == False:
        obs = env.reset()
        for s in range(0,250):
            action = np.argmax(agent.get_q_values_target_model(obs))
            obs_next, reward, done, _ = env.step(action)
            agent.replay_memory.add_experience(obs,action, reward, obs_next,done)
            if done:
                break

def train_agent(NUM_EPISODES, EXPERIENCE_REPLAY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NN_NUM, AVG_EVERY_NUM, EPSILON, EPSILON_DECAY):
    collect_experiences()
    rewards = []
    for e in range(0,NUM_EPISODES):

        obs = env.reset()
        if e == 0 or e%AVG_EVERY_NUM == 0:
            if EPSILON != 0:
                EPSILON -= EPSILON_DECAY

            episode_return = 0
        for step in range(0,250):
            print("Episode number ",e)
            rand = random.random()
            if rand < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_q_values_target_model(obs))
            obs_next, reward, done, _ = env.step(action)

            if done:
                if step < 250:
                    reward = -10
                else:
                    reward = 10
            episode_return += reward
            agent.replay_memory.add_experience(obs,action, reward, obs_next,done)
            s,a,r,s_,d = agent.replay_memory.get_mini_batch(TRAIN_BATCH_SIZE)

            agent.update_network_minibatch(s,a,r,s_,d)
            if done:
                break
        if e == 0 or e%AVG_EVERY_NUM == 0:
            rewards.append(episode_return/AVG_EVERY_NUM)
        if e%UPDATE_TARGET_NN_NUM == 0:
            agent.update_target_network()
    return rewards


env = gym.make('CartPole-v0')
action_space = env.action_space.n
obs_space = 4
agent = Agent.DQN_Agent(replay_size = REPLAY_MEMORY_SIZE,obs_space = obs_space, action_space = action_space, epsilon=EPSILON, gamma = DISCOUNT, alpha=LEARNING_RATE, activation_function = ['tanh'],hidden_layers_dim=HIDDEN_LAYERS_DIM)

rewards = train_agent(NUM_EPISODES,REPLAY_MEMORY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NUM, AVG_EVERY_NUM , EPSILON, EPSILON_DECAY)
print(rewards)
plt.plot(rewards)
plt.show()
done = False
obs = env.reset()
EPSILON = 0
r= 0
while not done:
    env.render()
    action = np.argmax(agent.get_q_values_target_model(obs))
    obs_next, reward, done, _ = env.step(action)
    r += reward
    obs=obs_next
print(r)

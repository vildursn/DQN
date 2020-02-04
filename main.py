import Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# 100000 episoder, 1000 replay memory size, 100 batch size tar over ent ime....
NUM_EPISODES = 100000
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPSILON = 0.3
REPLAY_MEMORY_SIZE = 1000
TRAIN_BATCH_SIZE = 100
UPDATE_TARGET_NUM = 10
HIDDEN_LAYERS_DIM = [10,10,10,10]

DISC_SPACES=[12,2,12,2]#cart pos, cart vel, stick angle, stick velocity
DISC_OBS_SPACE = DISC_SPACES[0]+DISC_SPACES[1]+DISC_SPACES[2]+DISC_SPACES[3]

#USE THE TARGET MODEL FOR ACTION SELECTION, AND THE REGULAR MODEL FOR ACTION EVALUATION

def cont_to_disc_obs(obs):
    disc_obs =np.zeros(DISC_OBS_SPACE)
    x = obs[0]
    x+=4.8
    for i in range(0,DISC_SPACES[0]):
        if x<=((4.8*2)/DISC_SPACES[0])*i:
            disc_obs[i]=1
            break
    x=obs[1]
    if x >0:
        disc_obs[DISC_SPACES[0]]=1
    else:
        disc_obs[DISC_SPACES[0]+1]=1

    x= obs[2]
    x+=0.42
    for i in range(0,DISC_SPACES[2]):
        if x<=((0.42*2)/DISC_SPACES[2])*i:
            disc_obs[DISC_SPACES[0]+DISC_SPACES[1]+i]=1
            break
    x=obs[3]
    if x >0:
        disc_obs[DISC_SPACES[0]+DISC_SPACES[1]+DISC_SPACES[2]]=1
    else:
        disc_obs[DISC_SPACES[0]+DISC_SPACES[1]+DISC_SPACES[2]+1]=1
    return disc_obs

def obs_to_nn(obs):
    return [[cont_to_disc_obs(obs)]]

def collect_experiences():
    while agent.replay_memory.is_replay_memory_full() == False:
        obs = env.reset()
        for s in range(0,250):
            action = np.argmax(agent.get_q_values_target_model(obs))
            obs_next, reward, done, _ = env.step(action)
            agent.replay_memory.add_experience(cont_to_disc_obs(obs),action, reward, cont_to_disc_obs(obs_next),done)
            if done:
                break

def train_agent(NUM_EPISODES, EXPERIENCE_REPLAY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NN_NUM, AVG_EVERY_NUM):
    collect_experiences()
    rewards = []
    for e in range(0,NUM_EPISODES):
        print(e)
        obs = env.reset()
        if e == 0 or e%AVG_EVERY_NUM == 0:
            r_ = 0
        for step in range(0,250):
            rand = random.random()
            if rand < EPSILON:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_q_values_target_model(obs))
            obs_next, reward, done, _ = env.step(action)
            r_+= reward
            if done and step < 250:
                reward = -1
            agent.replay_memory.add_experience(cont_to_disc_obs(obs),action, reward, cont_to_disc_obs(obs_next),done)
            s,a,r,s_,d = agent.replay_memory.get_mini_batch(TRAIN_BATCH_SIZE)

            agent.update_network_minibatch(s,a,r,s_,d)
            if done:
                break
        if e == 0 or e%AVG_EVERY_NUM == 0:
            rewards.append(r_/AVG_EVERY_NUM)
        if e%UPDATE_TARGET_NN_NUM == 0:
            agent.update_target_network()
    return rewards


env = gym.make('CartPole-v0')
action_space = env.action_space.n
agent = Agent.DQN_Agent(replay_size = REPLAY_MEMORY_SIZE,obs_space = DISC_OBS_SPACE, action_space = action_space, epsilon=EPSILON, gamma = DISCOUNT, alpha=LEARNING_RATE, activation_function = ['tanh'],DISC_SPACES=DISC_SPACES,hidden_layers_dim=HIDDEN_LAYERS_DIM)

rewards = train_agent(NUM_EPISODES,REPLAY_MEMORY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NUM, AVG_EVERY_NUM = 10)
print(rewards)
plt.plot(rewards)
plt.show()
done = False
obs = env.reset()
while not done:
    env.render()
    action = np.argmax(agent.get_q_values_target_model(obs))
    obs_next, reward, done, _ = env.step(action)
    obs=obs_next

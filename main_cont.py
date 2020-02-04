import Agent_cont as Agent
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# 100000 episoder, 1000 replay memory size, 100 batch size tar over ent ime....
NUM_EPISODES = 100
LEARNING_RATE = 0.01
DISCOUNT = 0.9
EPSILON = 0.4
EPSILON_DECAY = 0.002
REPLAY_MEMORY_SIZE = 100
TRAIN_BATCH_SIZE = 10
UPDATE_TARGET_NUM = 10
HIDDEN_LAYERS_DIM = [100,100,100]
AVG_EVERY_NUM = 1


#USE THE TARGET MODEL FOR ACTION SELECTION, AND THE REGULAR MODEL FOR ACTION EVALUATION

def epsilon_decay(e,epsilon):
    if e < NUM_EPISODES/2:
        return epsilon
    elif e < NUM_EPISODES*0.75:
        return 0.2
    elif e < NUM_EPISODES*0.9:
        return 0.1
    else:
        return 0

def collect_experiences():
    while agent.replay_memory.is_replay_memory_full() == False:
        obs = env.reset()
        for s in range(0,250):
            action = env.action_space.sample()
            obs_next, reward, done, _ = env.step(action)
            if done:
                if s < 250:
                    reward = -1
                else:
                    reward = 10
            agent.replay_memory.add_experience(obs,action, reward, obs_next,done)
            if done:
                break

def train_agent(agent,NUM_EPISODES, EXPERIENCE_REPLAY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NN_NUM, AVG_EVERY_NUM, EPSILON, EPSILON_DECAY):
    collect_experiences()
    rewards = []
    epsilon = EPSILON
    for e in range(0,NUM_EPISODES):

        obs = env.reset()
        epsilon = epsilon_decay(e,epsilon)
        #epsilon = 0.5
        episode_return = 0
        for step in range(0,250):
            print("Episode number ",e)
            rand = random.random()
            if rand <epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_q_values_target_model(obs))
            obs_next, reward, done, _ = env.step(action)

            if done:
                if step < 250:
                    reward = -1
                else:
                    reward = 10
            episode_return += reward
            agent.replay_memory.add_experience(obs,action, reward, obs_next,done)
            s,a,r,s_,d = agent.replay_memory.get_mini_batch(TRAIN_BATCH_SIZE)

            agent.update_network_minibatch(s,a,r,s_,d)
            if done:
                break
        rewards.append(episode_return)
        if e%UPDATE_TARGET_NN_NUM == 0:
            agent.update_target_network()
    return agent,rewards


env = gym.make('CartPole-v0')
action_space = env.action_space.n
obs_space = 4
agent = Agent.DQN_Agent(replay_size = REPLAY_MEMORY_SIZE,obs_space = obs_space, action_space = action_space, epsilon=EPSILON, gamma = DISCOUNT, alpha=LEARNING_RATE, activation_function = ['tanh'],hidden_layers_dim=HIDDEN_LAYERS_DIM)

agent,rewards = train_agent(agent,NUM_EPISODES,REPLAY_MEMORY_SIZE, TRAIN_BATCH_SIZE, UPDATE_TARGET_NUM, AVG_EVERY_NUM , EPSILON, EPSILON_DECAY)
print(rewards)
plt.plot(rewards)
plt.show()
done = False
obs = env.reset()
EPSILON = 0
r_tot= []

for e in range(0,100):
    r=0
    obs = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.get_q_values_target_model(obs))
        obs_next, reward, done, _ = env.step(action)
        r += reward
        obs=obs_next
    r_tot.append(r)

obs = env.reset()
r=0
done = False
while not done:
    env.render()
    action = np.argmax(agent.get_q_values_target_model(obs))
    obs_next, reward, done, _ = env.step(action)
    r += reward
    obs=obs_next
r_tot.append(r)
print(np.average(r_tot), r_tot)
agent.save_target_model()

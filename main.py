import Agent
import numpy as np
import gym
import tensorflow as tf

NUM_EPISODES = 1_000
LEARNING_RATE = 0.99
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50
MIN_REPLAY_MEMORY_SIZE = 1_000
TRAIN_BATCH_SIZE = 100
UPDATE_TARGET_NUM = 10

DISC_SPACES=[12,2,12,2]#cart pos, cart vel, stick angle, stick velocity
DISC_OBS_SPACE = DISC_SPACES[0]+DISC_SPACES[1]+DISC_SPACES[2]+DISC_SPACES[3]

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




env = gym.make('CartPole-v0')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = Agent.DQN_Agent(D_size = REPLAY_MEMORY_SIZE,obs_space = DISC_OBS_SPACE, action_space = action_space, epsilon=0.5, gamma = DISCOUNT, alpha=LEARNING_RATE, activation_function = ['tanh'],hidden_layers_dim=[10,10,10])
print("---> ", DISC_OBS_SPACE)

obs = env.reset()
disc_obs = cont_to_disc_obs(obs)
print(obs," == ", disc_obs, disc_obs.shape)
print(np.shape(disc_obs))

print("SAMPLE : ",env.action_space.sample(), env.observation_space.sample())
print("AGENT :",agent.get_action(disc_obs))
#for _ in range(1000):
#    env.render()
    #obs =env.step(agent.get_action(obs))
    #env.step(env.action_space.sample()) # take a random action
env.close()

ep_rewards = []

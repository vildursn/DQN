import Agent
import numpy as np
import gym
import tensorflow as tf

NUM_EPISODES = 1_000
LEARNING_RATE = 0.99
DISCOUNT = 0.99
EPSILON = 0.5
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


def obs_to_nn(obs):
    return [[cont_to_disc_obs(obs)]]

env = gym.make('CartPole-v0')
#obs_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = Agent.DQN_Agent(D_size = REPLAY_MEMORY_SIZE,obs_space = DISC_OBS_SPACE, action_space = action_space, epsilon=0.5, gamma = DISCOUNT, alpha=LEARNING_RATE, activation_function = ['tanh'],DISC_SPACES=[12,2,12,2],hidden_layers_dim=[10,10,10])
test=True
obs= tf.Variable(tf.zeros([DISC_OBS_SPACE]))
if test:
    agent.print_replay_memory()
    for i in range(0,50):
        obs = env.reset()
        #print(obs)
        action = agent.get_action(obs)
        #print(action)
        returned = env.step(action)
        #print(returned)
        obs_next, reward, done, _ = returned
        #print(obs_next, reward, done)
        agent.update_replay_memory([obs,action, reward, obs_next, done])

    agent.print_replay_memory()
    #agent.update_network(obs, action, reward, obs_next, done)
    #print([[obs]])
    #print(np.shape([[obs]]))
    #action = agent.get_action(obs)
    #print(action)
    #action=agent.get_action([[cont_to_disc_obs(obs),cont_to_disc_obs(obs)]])
    #print(action)
    #returned = env.step(action)
    #obs_next= cont_to_disc_obs(returned[0])
    #reward = returned[1]
    #tot_Reward+=reward
    #d=returned[2]
    #agent.update_network(obs,action, reward, obs_next,d)
    #print(action)
else:
    rewardz = []
    for e in range(0, NUM_EPISODES):
        tot_Reward=0
        obs = env.reset()
        obs=cont_to_disc_obs(obs)
        #print(shape(obs))
        for step in range(0,500):

            if EPSILON > np.random.rand():
                action =agent.get_action([[obs]])
            else:
                action = env.action_space.sample()
            #print(obs)
            #env.render()
            returned = env.step(action)
            obs_next= cont_to_disc_obs(returned[0])
            reward = returned[1]
            tot_Reward+=reward
            d=returned[2]
            #sars_d = obs[0][0]
            #print("hieheieie",obs[0][0])
            #sars_d = np.append(sars_d,action)
            #sars_d = append(action)
            #sars_d.append(reward)
            #sars_d = np.append(sars_d,reward)
            #sars_d.append(obs_next[0][0])
            #sars_d = np.append(sars_d,obs_next[0][0])
            #sars_d = np.append(sars_d,d)

            #sars_d.append(d)
            #print("----", sars_d)
            #print("----------------------",type(sars_d[0]))
            #print("----------------------",type(sars_d[0][0]))
            #agent.update_replay_memory(sars_d)
            agent.update_replay_memory([obs,action,reward,obs_next, d])
            agent.update_network()
            obs = obs_next
            if d :
                break
        rewardz.append(tot_Reward)

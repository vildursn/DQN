
import numpy as np
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense


DISC_SPACES=[12,2,12,2]#cart pos, cart vel, stick angle, stick velocity
DISC_OBS_SPACE = DISC_SPACES[0]+DISC_SPACES[1]+DISC_SPACES[2]+DISC_SPACES[3]
BATCH_SIZE=5


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
def createNN():
    model = Sequential()
    #print("!!!!!", self.obs_space)
    model.add(Dense(10, input_dim= (4)))
    for i in range(1,4):
        model.add(Dense(10, activation = 'tanh'))
    model.add(Dense(ACTION_SPACE, activation = tf.nn.sigmoid))
    model.compile(loss='mse', optimizer='sgd')#learning_rate = self.alpha,
    return model

class Q_Network():
    def __init__(self, input_dim, hidden_layers_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_layers_dim=hidden_layers_dim
        self.output_dim = output_dim
        self.model = self.build_nn()

    def build_nn(self):
        model = Sequential()
        model.add(Dense(10,batch_input_shape=(None,DISC_OBS_SPACE)))



env = gym.make('CartPole-v0')
ACTION_SPACE = env.action_space.n
model = createNN()
test = True
if test:
    obs = env.reset()
    states = tf.placeholder(tf.int32, shape(BATCH_SIZE, DISC_OBS_SPACE), name='state')
    #print(type(obs),np.shape(obs),obs)
    #d_obs = cont_to_disc_obs(obs)
    #print(type(d_obs),np.shape(d_obs),d_obs)
    #nn_obs = obs_to_nn(obs)
    #print(type(nn_obs),np.shape(nn_obs),nn_obs)
    #obs2 = env.reset()
    #observation_list = list()
    #observation_list.append(obs)
    #observation_list.append(obs2)
    action=np.argmax(model.predict(obs))
    print(action)
else:
    obs = env.reset()
    print(obs)
    action=np.argmax(model.predict(obs))#, batch_size=1)

    returned = env.step(action)
    obs_next, reward, done, _ = returned[0],returned[1],returned[2],returned[3]
    obs2 = env.reset()
    q_val = np.max(model.predict(obs))
    model.fit(obs, q_val)
    print(q_val)

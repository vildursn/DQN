import numpy as np
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.callbacks import TensorBoard
import time
import random


class replay_memory():
    def __init__(self, size, obs_space,action_space):
        self.size = size
        self.obs_space = obs_space
        self.action_space = action_space
        self.obs = np.zeros((size, obs_space))
        self.actions = np.zeros(size)
        self.rewards = np.zeros(size)
        self.obs_nexts = np.zeros((size,obs_space))
        self.done = np.zeros(size)
        self.counter = 0
        self.replay_memory_full = False

    def add_experience(self, s, a ,r, s_, d):
        self.obs[self.counter] = s
        self.actions[self.counter] = a
        self.rewards[self.counter] = r
        self.obs_nexts[self.counter]=s_
        self.done[self.counter] = d
        self.counter +=1
        if self.counter == self.size :
            self.replay_memory_full = True
            self.counter =0
    def is_replay_memory_full(self):
        return self.replay_memory_full
    def print_replay_memory(self):
        if self.replay_memory_full:
            for i in range(0,self.size):
                print(i,self.obs[i],self.actions[i],self.rewards[i],self.obs_nexts[i],self.done[i])
        else:
            for i in range(0,self.counter):
                print(i,self.obs[i],self.actions[i],self.rewards[i],self.obs_nexts[i],self.done[i])

    def get_mini_batch(self, mini_batch_size):
        if mini_batch_size > self.size:
            return False
        else:
            if self.replay_memory_full == False :
                return False
            else:
                numbers = random.sample(range(0,self.size), mini_batch_size)
                s = np.zeros((mini_batch_size,self.obs_space))
                a = np.zeros(mini_batch_size)
                r = np.zeros(mini_batch_size)
                s_ = np.zeros((mini_batch_size,self.obs_space))
                d = np.zeros(mini_batch_size)
                for i in range(0,mini_batch_size):
                    s[i]=self.obs[i]
                    a[i]=self.actions[i]
                    r[i]=self.rewards[i]
                    s_[i]=self.obs_nexts[i]
                    d[i]=self.done[i]
        return s,a,r,s_,d


class DQN_Agent():
    #self.Q_function = Q_function()
    #self.replay_memory = Replay
#obs space and action space just wants their lengths, and they are all assumed to be discrete?
    def __init__(self, model):
        self.target_model = model

    def __init__(self,replay_size,obs_space, action_space, epsilon, gamma, alpha, activation_function,hidden_layers_dim):

        self.target_update_counter = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha #learning learning_rate
        self.obs_space = obs_space
        self.action_space = action_space
        self.hidden_layers_dim = hidden_layers_dim
        self.activation_function = activation_function
        #datatype=[tf.int64]*(self.obs_space*2+2)
        #datatype.append(tf.bool)
        #self.replay_memory = tf.queue.FIFOQueue(REPLAY_MEMORY_SIZE, dtypes=datatype)#, shape=[obs_space,1,1,obs_space,1], shape =  [self.obs_space, self.action_space,1, self.obs_space])
        #self.replay_memory = np.array((50,5))
        #self.replay_memory_counter = 0
        #self.replay_memory_full = False
        self.replay_memory = replay_memory(replay_size,obs_space,action_space)

        self.model = self.createNN()
        self.target_model = self.createNN()
        self.target_model.set_weights(self.model.get_weights())

        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format('MODEL_NAME', int(time.time())))

    def createNN(self):
        model = Sequential()
        #print("!!!!!", self.obs_space)
        model.add(Dense(self.hidden_layers_dim[0], input_dim=self.obs_space+self.action_space))
        for i in range(1,len(self.hidden_layers_dim)):
            if len(self.hidden_layers_dim)== len(self.activation_function): #If there are different activation functions for different hidden layers
                model.add(Dense(self.hidden_layers_dim[i], activation = self.activation_function[i]))
            else:
                model.add(Dense(self.hidden_layers_dim[i], activation = self.activation_function[0]))
        model.add(Dense(1, activation = None))
        model.compile(loss='mse', optimizer='sgd')#learning_rate = self.alpha,
        return model

    def merge_obs_action(self, obs, action):
        actions = np.zeros(self.action_space)
        for i in range(0, self.action_space):
            if action == i:
                actions[i]=1
                break
        obs = np.append(obs,actions)
        return obs

    def get_q_values_model(self, obs):
        #print(self.action_space)
        predictions = np.zeros(self.action_space)
        for i in range(0, self.action_space):
            input = self.merge_obs_action(obs,i)
            #print(input)
            #print(np.shape(input))
            predictions[i] = self.model.predict([[input]])
        return predictions
    def get_q_values_target_model(self, obs):
        #print(self.action_space)
        predictions = np.zeros(self.action_space)
        for i in range(0, self.action_space):
            input =self.merge_obs_action(obs,i)
            #print(np.shape(input))
            #print(input)
            predictions[i] = self.target_model.predict([[input]])
        return predictions


    def get_action_target_model(self, obs, action):
        return np.argmax(self.target_model.predict(self.merge_obs_action(obs,i)))

    def get_best_action_model(self, obs):
        return np.argmax(agent.get_q_values(obs))

    def update_replay_memory(self,sars_d):

        if self.replay_memory_counter == REPLAY_MEMORY_SIZE:
            if self.replay_memory_full == False:
                self.replay_memory_full = True
            self.replay_memory_counter = 0
        self.replay_memory[self.replay_memory_counter] = sars_d
        self.replay_memory_counter += 1




    def update_network_minibatch(self,s,a,r,s_,d):
        inputs = np.zeros((len(a),self.obs_space+self.action_space))
        y = np.zeros(len(a))
        for i in range(0,len(a)):
            inputs[i] = self.merge_obs_action(s[i],a[i])
            if d[i] :
                y[i]=r[i]
            else:
                y[i] = r[i] + self.gamma*np.max(self.get_q_values_model(s_[i]))
        self.model.fit(inputs,y, batch_size = len(a),epochs = 10, verbose = 0)

    def update_target_network(self):
        t_w = np.array(self.target_model.get_weights())
        m_w = np.array(self.model.get_weights())
        new_weights = 0.1*t_w + 0.9*m_w
        self.target_model.set_weights(new_weights)

    def save_target_model(self):
        self.target_model.save("Target_model040220_2")
        print("Model saved.")

        #self.target_model.set_weights(self.model.get_weights()*0.9 + self.target_model.get_weights()*0.1)


class Pretrained_Agent(DQN_Agent):
    def __init__(self,model, action_space):
        self.target_model = model
        self.action_space = action_space
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

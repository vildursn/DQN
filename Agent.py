import numpy as np
import tensorflow as tf
import keras.models
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.callbacks import TensorBoard
import time
import random

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50
MIN_REPLAY_MEMORY_SIZE = 1_000
MINI_BATCH_SIZE = 100
UPDATE_TARGET_NUM = 10

class replay_memory():
    def __init__(self, size):
        pass
class DQN_Agent():
    #self.Q_function = Q_function()
    #self.replay_memory = Replay
#obs space and action space just wants their lengths, and they are all assumed to be discrete?
    def __init__(self,D_size,obs_space, action_space, epsilon, gamma, alpha, activation_function,hidden_layers_dim):
        self.d_size = D_size
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
        self.replay_memory = []

        self.model = self.createNN()
        self.target_model = self.createNN()
        self.target_model.set_weights(self.model.get_weights())

        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format('MODEL_NAME', int(time.time())))

    def createNN(self):
        model = Sequential()
        #print("!!!!!", self.obs_space)
        model.add(Dense(self.hidden_layers_dim[0], input_dim=self.obs_space))
        for i in range(1,len(self.hidden_layers_dim)):
            if len(self.hidden_layers_dim)== len(self.activation_function): #If there are different activation functions for different hidden layers
                model.add(Dense(self.hidden_layers_dim[i], activation = self.activation_function[i]))
            else:
                model.add(Dense(self.hidden_layers_dim[i], activation = self.activation_function[0]))
        model.add(Dense(self.action_space, activation = tf.nn.sigmoid))
        model.compile(loss='mse', optimizer='sgd')#learning_rate = self.alpha,
        return model

    def get_q_values(self, obs):
        return self.model.predict(obs)

    def get_action(self, obs):
        #print(obs)
        #print(np.shape(obs))
        return np.argmax(self.target_model.predict(obs))

    def update_replay_memory(self,sars_d):
        if len(self.replay_memory)>= REPLAY_MEMORY_SIZE:
            #print(type(self.replay_memory))
            self.replay_memory = np.delete(self.replay_memory,0)
        self.replay_memory = np.append(self.replay_memory,sars_d)


    def __update_replay_memory(self,sars_d):
        if self.replay_memory.size() == REPLAY_MEMORY_SIZE:
            self.replay_memory.dequeue()
        self.replay_memory.enqueue(sars_d)



    def sample_random_situations(self):
        nums = random.sample(range(0,MIN_REPLAY_MEMORY_SIZE),MINI_BATCH_SIZE)
        mini_batch=[None]*MINI_BATCH_SIZE
        for i in range(0,MINI_BATCH_SIZE):
            mini_batch[i]=self.replay_memory[nums[i]]
        return mini_batch

    def update_network(self, s,a,r,s_,d):
        if d:
            y = r
        else:
            y = r + self.gamma*np.max(self.get_q_values(s_))
        self.model.fit(s,y)

    def update_network_minibatch(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return None
        else:
            y=np.zeros(len(mini_batch))
            mini_batch = self.sample_random_situations()
            for i in range(0,len(mini_batch)):
                if mini_batch[i][4]:
                    y[i]=mini_batch[i][2]
                else:
                    y[i]=mini_batch[i][2] + self.gamma*np.max(self.target_model.predict(mini_batch[3]))
                    ###EKSTRA BRAKCET?
            self.model.fit(mini_batch[0],y, batch_size = TRAIN_BATCH_SIZE, verbose=0, shuffle=False)





    def __update_network(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return None
        else:
            #randomly sample from the replay_memory
            full_list = self.replay_memory
            batch =np.array(TRAIN_BATCH_SIZE)
            for i in range(0,TRAIN_BATCH_SIZE):
                r = np.random.randint(0,len(self.replay_memory)-i)
                batch[i]=full_list[r]
                full_list = np.delete(full_list,r)
            #batch = random.sample(list(self.replay_memory),TRAIN_BATCH_SIZE )
            #print("BATCH[0] = ",batch.shape())
            print(batch)
            states = np.array([situation[0] for situation in batch])
            next_states = np.array([situation[3] for situation in batch])

            next_states_q_target_val = self.target_model.predict(next_states)
            states_q_val = self.target_model.predict(states)

            actions = np.array([situation[1] for situation in batch])
            rewards = np.array([situation[2] for situation in batch])
            terminal_states_or_not = np.array([situation[3]] for situation in batch)

            #### BATCH SIZ UPDATING REPLAY EMMORY NOT RANDOM?????? Why every term epsiode only counting updating every target ....
            X =[]
            y=[]
            for it in range(0, len(batch)):
                if done[it]:
                    new_q = rewards[it]
                else:
                    max_next_state_q_val = np.max(next_states_q_target_val[it])
                    new_q =rewards[it] + self.gamma*max_next_state_q_val

                current_q_values = states_q_val[it]
                current_q_values[action[it]]= new_q

                X.append(states[it])
                y.append(current_q_values)

            self.model.fit(X,y, batch_size = TRAIN_BATCH_SIZE, verbose=0, shuffle=False, callbacks=([self.tensorboard] if terminal_states_or_not else None))
            if terminal_states_or_not:
                self.target_update_counter +=1
            if self.target_update_counter > UPDATE_TARGET_NUM:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0


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

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard

class DQN_Agent():
    #self.Q_function = Q_function()
    #self.replay_memory = Replay
    def __init__(self, D_size,obs_space, action_space, epsilon, gamma, alpha, activation_function,hidden_layers_dim):
        self.replay_memory = tf.queue.QueueBase.FIFOQueue(capacity = D_size, dtypes=[float, int, float, float], shape =  [len(self.obs_space), len(self.action_space),1, len(self.obs_space)])
        self.target_update_counter = 0
        self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha #learning rate
        self.obs_space = obs_space
		self.action_space = action_space
        self.hidden_layers_dim = hidden_layers_dim
        self.activation_function = activation_function
        self.model = self.Q_function()
        self.target_model = self.Q_function()
        self.target_model.set_weights(self.model.get_weights())
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def createNN(self):
    		model = tf.keras.models.Sequential()
    		model.add(Dense(self.hidden_layers_dim[0], input_shape=(self.obs_space,)))
    		for i in range(1,len(self.hidden_layers_dim)):
    			if len(self.activation_function) == len(self.hidden_layers_dim): #If there are different activation functions for different hidden layers
    				model.add(Dense(self.hidden_layers_dim[i]), activation = self.activation_function[i])
    			else:
    				model.add(Dense(self.hidden_layers_dim[i], activation = self.activation_function))
    		model.add(Dense(self.action_space, activation = tf.nn.sigmoid))
            model.compile(loss='mse', optimizer='sgd', learning_rate = self.alpha)
    		return model


    def update_replay_memory(self,sars):
        self.replay_memory.enqueue(sars)



class Q_function():
    def __init__(self, obs_space, action_space, epsilon, gamma, alpha, hidden_layers_dim):

		self.policy_function = self.createNN(input_dim = obs_space, hidden_layers_dim= hidden_layers_dim,output_dim = action_space, alpha = alpha)




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

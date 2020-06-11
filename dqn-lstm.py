'''
courtesy: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
'''
import csv
import itertools
import os
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import normalize
from tqdm import tqdm

from environment import Environment
from lib import plotting

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print("in gpu")
else:
    print("not in gpu")

DISCOUNT = 0.90
REPLAY_MEMORY_SIZE = 10000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1050  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 1024  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 50  # Terminal states (end of episodes)
MODEL_NAME = 'dqn-lstm'
MIN_REWARD = -500  # For model save
MEMORY_FRACTION = 0.20
LEARNING_RATE = 0.001

# Environment settings
EPISODES = 12000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.05

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

env = Environment()

# For stats
ep_rewards = [-1200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

filename = "data/state-trace.csv"
# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


def csv_writer(row):
    with open(filename, 'a+') as file:
        writer = csv.writer(file)
        for item in row:
            writer.writerow(eval(item))


# Own Tensorboard class
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


# Agent class
class DQNLSTMAgent:
    def __init__(self, state_size, action_size, model_file='models/' + MODEL_NAME + '.model'):
        self.n_states = state_size
        self.n_actions = action_size

        if os.path.exists(model_file):
            self.model = load_model(model_file)
            self.target_model = load_model(model_file)
        else:
            # Main model
            self.model = self.create_model()

            # Target network
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(500, activation='relu', input_shape=(1, 7)))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))  # linear or softmax
        # model = model(states)

        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=LEARNING_RATE), metrics=['accuracy'])
        # print(model.summary())
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.zeros((len(minibatch), self.n_states))
        new_current_states = np.zeros((len(minibatch), self.n_states))

        for i, transition in enumerate(minibatch):
            current_states[i] = transition[0]
            new_current_states[i] = transition[3]

        # convert to 3D array to get prediction from LSTM module
        current_states = np.reshape(current_states, (len(current_states), 1, self.n_states))
        new_current_states = np.reshape(new_current_states, (len(new_current_states), 1, self.n_states))

        current_qs_list = self.model.predict(current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        # convert to 2D array for iterating over the values
        current_qs_list = np.reshape(current_qs_list, (len(current_qs_list), self.n_actions))
        future_qs_list = np.reshape(future_qs_list, (len(future_qs_list), self.n_actions))

        X = np.zeros((len(minibatch), self.n_states))
        y = np.zeros((len(minibatch), self.n_actions))

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X[index] = current_state
            y[index] = current_qs

        # reshape X and y to compatible with LSTM [batch size, time step, sequence]
        X = np.reshape(X, (MINIBATCH_SIZE, 1, self.n_states))
        y = np.reshape(y, (MINIBATCH_SIZE, 1, self.n_actions))
        # print("Training X: ", X)
        # print("Training y: ", y.shape)
        # X = np.array(X).reshape(64, 7)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(X, y, batch_size=MINIBATCH_SIZE, epochs=10, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        # print("State for Prediction: ", state)
        # make 3-D array for LSTM [batch, time step, sequence]
        state = np.reshape(state, (1, 1, self.n_states))
        # print("Reshaped State for Prediction: ", state)
        predict = self.model.predict(state)
        # print(predict)
        return predict


# RUN program starts here.
agent = DQNLSTMAgent(7, 3)
# print(agent.model.summary())
# Iterate over episodes

stats = plotting.EpisodeStats(
    episode_lengths=np.zeros(EPISODES + 1),
    episode_rewards=np.zeros(EPISODES + 1))

# step_actions = []
total_step = 1
# total_states = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    # print(current_state)
    # total_states.append(str(current_state))

    current_state = np.asarray(current_state)

    current_state = normalize(current_state)
    current_state = current_state.reshape(1, 7)

    # Reset flag and start iterating until episode ends
    done = False
    for t in itertools.count():

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
            # print(action)
        else:
            # Get random action
            action = np.random.randint(0, env.n_actions)

        new_state, reward, done = env.step(action)
        # if not done:
        #    total_states.append(str(new_state))

        # print(current_state, action, reward)

        new_state = np.asarray(new_state)
        new_state = normalize(new_state)
        new_state = new_state.reshape(1, 7)

        episode_reward += reward

        stats.episode_rewards[episode] += reward
        stats.episode_lengths[episode] = t

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        total_step += 1
        if done:
            break

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

        # # Save model, but only when min reward is greater or equal a set value
        # if min_reward >= MIN_REWARD:
        #     agent.model.save(f'models/{MODEL_NAME}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.model.save(f'models/{MODEL_NAME}.model')
print("Total Steps: ", total_step)
print("Missed deadline: ", env.missed_deadline)
# print(total_states)
# csv_writer(total_states)

plotting.plot_episode_stats(stats, filename="dqn-lstm")

import random
import time

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from environment import Environment
from lib import plotting
import itertools

# Exploration settings
MAX_EPSILON = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.05
GAMMA = 0.9
LEARNING_RATE = 0.001
MODEL_NAME = "DQN-TF"
ep_rewards = []
AGGREGATE_STATS_EVERY = 10  # episodes
EPISODES = 10000

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


# Own Tensorboard class
class ModTensorBoard(TensorBoard):

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


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()
        self.tensorboard = ModTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def _define_model(self):
        self._states = tf.compat.v1.placeholder(shape=[None, self._num_states], dtype=tf.float64)
        self._q_s_a = tf.compat.v1.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.compat.v1.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
        self._var_init = tf.compat.v1.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=False):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        for episode in range(EPISODES):
            self._model.tensorboard.step = episode
            state = self._env.reset()
            state = np.asarray(state)
            tot_reward = 0
            max_x = -100

            env.total_cost = 0
            env.exe_delay = 0
            env.tot_energy_cost = 0
            env.tot_off_cost = 0
            env.off_decisions = {0: 0, 1: 0, 2: 0}
            env.off_from_edge = 0

            for step in itertools.count():
                if self._render:
                    self._env.render()

                action = self._choose_action(state)
                next_state, reward, done = self._env.step(action)
                next_state = np.asarray(next_state)

                stats.episode_rewards[episode] += reward
                stats.episode_lengths[episode] = step

                # print(state,action, reward)
                # None for storage sake
                if done:
                    next_state = None

                self._memory.add_sample((state, action, reward, next_state))
                self._replay()

                # exponentially decay the eps value
                self._steps += 1
                # if self._eps > MIN_EPSILON:
                #     self._eps *= EPSILON_DECAY
                #     self._eps = max(MIN_EPSILON, self._eps)

                # self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.math.exp(-EPSILON_DECAY * self._steps)

                # move the agent to the next state and accumulate the reward
                state = next_state
                tot_reward += reward

                # if the game is done, break the loop
                if done:
                    self._reward_store.append(tot_reward)
                    self._max_x_store.append(max_x)
                    break

            if self._eps > MIN_EPSILON:
                self._eps *= EPSILON_DECAY
                self._eps = max(MIN_EPSILON, self._eps)

            ep_rewards.append(tot_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                self._model.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                     reward_max=max_reward, reward_per_ep=tot_reward,
                                                     epsilon=self._eps)

            print("Episode {}, Step {}, Total reward: {}, Eps: {}".format(episode, self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            q_values = self._model.predict_one(state, self._sess)
            action = np.argmax(q_values)
            return action

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)


if __name__ == "__main__":
    # env_name = 'MountainCar-v0'
    # env = gym.make(env_name)
    env = Environment()

    num_states = 7
    num_actions = 3
    num_episodes = 10000

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    model = Model(num_states, num_actions, 128)
    mem = Memory(100000)

    with tf.Session() as sess:
        sess.run(model._var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        EPSILON_DECAY)
        # cnt = 0
        #         # while cnt < num_episodes:
        #         #     model.tensorboard.step = cnt
        #         #     if cnt % 10 == 0:
        #         #         print('Episode {} of {}'.format(cnt + 1, num_episodes))
        #         #     gr.run(cnt)
        #         #     cnt += 1
        gr.run()
        # saver = tf.train.Saver()
        # saver.save(sess, "models/dqn_tf")
        # plt.plot(gr._reward_store)
        # plt.show()
        # plt.close("all")

        print("Total Costs: ", env.total_cost)
        print("Total Execution Time: ", env.exe_delay)
        print("Total Energy cost: ", env.tot_energy_cost)
        print("Total Money for offloading: ", env.tot_off_cost)
        print("Offloading numbers", env.off_decisions)
        print("offload from edge: ", env.off_from_edge)
        # plotting.plot_episode_stats(stats, filename="dqn-tf-lr-0.0001-b1024-rm-10k")
        # plt.plot(gr._max_x_store)
        plt.show()

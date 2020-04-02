import csv
import os
import shutil
import itertools
import matplotlib.style
import numpy as np
from lib import plotting

from environment import Environment
from qlearning import QLearningTable
from pandas import DataFrame

matplotlib.style.use('ggplot')


def save_state_actions(data):
    with open("data/state_action.csv", "a+") as state_action:
        csv_writer = csv.writer(state_action, delimiter=",")
        csv_writer.writerow(data)


def update(env, RL, episodes=1000):
    ep_no = 1
    # state = env.reset()
    for episode in range(episodes):
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (ep_no, t))

            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)

            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            # logging the state, action and reward
            # save_state_actions([str(state), action, reward])
            # update the q table.
            RL.learn(str(state), action, reward, str(state_))

            state = state_

            if done:
                break
        ep_no += 1
    save_to_file(RL.q_table)
    print("complete")
    print(stats)


def save_to_file(q_table):
    data_dir = "data/q_table.csv"
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    DataFrame.to_csv(q_table, data_dir)


if __name__ == '__main__':
    num_of_episodes = 5000

    env = Environment()

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_of_episodes),
        episode_rewards=np.zeros(num_of_episodes))

    RL = QLearningTable(actions=list(range(env.n_actions)))

    update(env, RL, episodes=num_of_episodes)

    plotting.plot_episode_stats(stats)

    # shutil.copyfile("data/q_table.csv", "data/episodes-" + str(500) + ".csv")
    #
    # for i in range(3, 11):
    #     iteration = i * 10000
    #     update(env, RL, episodes=iteration)
    #     shutil.copyfile("data/q_table.csv", "data/episodes-" + str(iteration) + ".csv")
    #     # os.rename("data/q_table.csv", "data/episodes-" + str(iteration) + ".csv")

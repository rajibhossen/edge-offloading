import csv
import os
import shutil
import itertools
import matplotlib.style
import numpy as np
from lib import plotting

from environment import Environment
from naive_env import NaiveEnvironment
from qlearning import QLearningTable
from pandas import DataFrame

matplotlib.style.use('ggplot')

# For stats
ep_rewards = []
AGGREGATE_STATS_EVERY = 50  # episodes

filename = "data/all-mobile-avg-reward.csv"
with open(filename, "a+") as avg_reward:
    csv_writer = csv.writer(avg_reward, delimiter=",")
    csv_writer.writerow(['Step', 'Value'])


def save_avg_reward(data):
    with open(filename, "a+") as avg_reward:
        csv_writer = csv.writer(avg_reward, delimiter=",")
        csv_writer.writerow(data)


def update(env, episodes=12000):
    # state = env.reset()
    for episode in range(1, episodes + 1):
        total_reward = 0
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (episode, t))
            action = 2
            state_, reward, done = env.step(action)
            # print(state,reward)
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            total_reward += reward
            state = state_
            if done:
                break
        ep_rewards.append(total_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            # save_avg_reward([episode, average_reward])
    # save_to_file(RL.q_table)
    print("Missed deadline: ", env.missed_deadline)
    print("Total Execution Time: ", env.exe_delay)
    print("Total Energy cost: ", env.tot_energy_cost)
    print("complete")
    # print(stats)


# def save_to_file(data):
#     data_dir = "data/all-mobile-avg-reward.csv"
#     # if not os.path.exists(data_dir):
#     #     os.mkdir(data_dir)
#     DataFrame.to_csv(data, data_dir)


if __name__ == '__main__':
    num_of_episodes = 12000

    # env = NaiveEnvironment()
    env = Environment()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_of_episodes + 1),
        episode_rewards=np.zeros(num_of_episodes + 1))

    # RL = QLearningTable(actions=list(range(env.n_actions)))

    update(env, episodes=num_of_episodes)

    plotting.plot_episode_stats(stats)

    # shutil.copyfile("data/q_table.csv", "data/episodes-" + str(500) + ".csv")
    #
    # for i in range(3, 11):
    #     iteration = i * 10000
    #     update(env, RL, episodes=iteration)
    #     shutil.copyfile("data/q_table.csv", "data/episodes-" + str(iteration) + ".csv")
    #     # os.rename("data/q_table.csv", "data/episodes-" + str(iteration) + ".csv")

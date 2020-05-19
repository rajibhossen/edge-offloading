import csv
import os
import shutil
import itertools
import matplotlib.style
import numpy as np

from cloud import Cloud
from edge import Edge
from lib import plotting

from environment import Environment
from mobile import Mobile
from naive_env import NaiveEnvironment
from qlearning import QLearningTable
from pandas import DataFrame

matplotlib.style.use('ggplot')

# For stats
mobile_costs = []
edge_costs = []
cloud_costs = []
naive_costs = []

AGGREGATE_STATS_EVERY = 50  # episodes

m_file = "data/all-mobile-avg-reward.csv"
with open(m_file, "a+") as avg_reward:
    csv_writer = csv.writer(avg_reward, delimiter=",")
    csv_writer.writerow(['Step', 'Value'])

e_file = "data/all-edge-avg-reward.csv"
with open(e_file, "a+") as avg_reward:
    csv_writer = csv.writer(avg_reward, delimiter=",")
    csv_writer.writerow(['Step', 'Value'])

c_file = "data/all-cloud-avg-reward.csv"
with open(c_file, "a+") as avg_reward:
    csv_writer = csv.writer(avg_reward, delimiter=",")
    csv_writer.writerow(['Step', 'Value'])

n_file = "data/naive_approach.csv"
with open(n_file, "a+") as avg_reward:
    csv_writer = csv.writer(avg_reward, delimiter=",")
    csv_writer.writerow(['Step', 'Value'])


def m_save(data):
    with open(m_file, "a+") as mr:
        csv_writer = csv.writer(mr, delimiter=",")
        csv_writer.writerow(data)


def e_save(data):
    with open(e_file, "a+") as er:
        csv_writer = csv.writer(er, delimiter=",")
        csv_writer.writerow(data)


def c_save(data):
    with open(c_file, "a+") as cr:
        csv_writer = csv.writer(cr, delimiter=",")
        csv_writer.writerow(data)


def n_save(data):
    with open(n_file, "a+") as nr:
        csv_writer = csv.writer(nr, delimiter=",")
        csv_writer.writerow(data)


def get_action(state):
    data = state[0]
    cpu_cycle = state[1]
    uplink_rate = state[3]
    mobile_cap = state[4]
    server_cap = state[5]
    energy_left = state[6]

    device = Mobile(mobile_cap)
    # energy_factor = energy_left / 100.0
    # energy_factor = 1 - energy_factor
    m_total, m_time, m_energy = device.calculate_cost_naive(cpu_cycle, 0.5)
    edge = Edge(uplink_rate, server_cap)
    e_total, e_time, e_energy = edge.cal_total_cost_naive(data, cpu_cycle, 0.5)
    cloud = Cloud(uplink_rate)
    c_total, c_time, c_energy = cloud.cal_total_cost_naive(data, cpu_cycle, 0.5)

    costs = [m_total, e_total, c_total]
    # get minimum cost action
    action = costs.index(min(costs))
    # print(costs)
    return -m_total, -e_total, -c_total, -min(costs), action


def update(env, episodes=4):
    # state = env.reset()
    for episode in range(1, episodes + 1):
        m_cost, e_cost, c_cost, n_cost = 0, 0, 0, 0
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (episode, t))

            # cost, action = get_action(state)
            mc, ec, cc, nc, action = get_action(state)
            state_, reward, done = env.step(action)
            # print(reward)

            m_cost += mc
            e_cost += ec
            c_cost += cc
            n_cost += nc

            state = state_
            if done:
                break

        mobile_costs.append(m_cost)
        edge_costs.append(e_cost)
        cloud_costs.append(c_cost)
        naive_costs.append(n_cost)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            m_avg = sum(mobile_costs[-AGGREGATE_STATS_EVERY:]) / len(mobile_costs[-AGGREGATE_STATS_EVERY:])
            m_save([episode, m_avg])

            e_avg = sum(edge_costs[-AGGREGATE_STATS_EVERY:]) / len(edge_costs[-AGGREGATE_STATS_EVERY:])
            e_save([episode, e_avg])

            c_avg = sum(cloud_costs[-AGGREGATE_STATS_EVERY:]) / len(cloud_costs[-AGGREGATE_STATS_EVERY:])
            c_save([episode, c_avg])

            n_avg = sum(naive_costs[-AGGREGATE_STATS_EVERY:]) / len(naive_costs[-AGGREGATE_STATS_EVERY:])
            n_save([episode, n_avg])

    # save_to_file(RL.q_table)
    print("complete")
    # print(stats)


if __name__ == '__main__':
    num_of_episodes = 40000

    env = Environment()

    update(env, episodes=num_of_episodes)

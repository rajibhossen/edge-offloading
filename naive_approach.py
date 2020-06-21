import itertools

import matplotlib.style

from cloud import Cloud
from edge import Edge
from environment import Environment
from generate_states import read_state_from_file
from mobile import Mobile

matplotlib.style.use('ggplot')
import numpy as np

# For stats


def calculate_costs(state):
    data = state[0]
    cpu_cycle = state[1]
    uplink_rate = state[2]
    mobile_cap = state[3]
    server_cap = state[4]
    energy_left = state[5]
    global mobile_energy, mobile_exec, edge_energy, edge_exec, cloud_energy, \
        cloud_exec, naive_energy, naive_exec, mobile_miss_dt, edge_miss_dt, \
        cloud_miss_dt, naive_miss_dt, edge_prices, cloud_prices, naive_prices

    device = Mobile(mobile_cap)
    m_total, m_time, m_energy = device.calculate_total_cost(cpu_cycle)
    edge = Edge(uplink_rate, server_cap)
    e_total, e_time, e_energy, e_price = edge.cal_total_cost(data, cpu_cycle)
    #print(e_time, e_energy, e_price)
    cloud = Cloud(uplink_rate)
    c_total, c_time, c_energy, c_price = cloud.cal_total_cost(data, cpu_cycle)
    #print(m_total, e_total, c_total)
    # naive greedy algorithms action choice
    costs = [m_total, e_total, c_total]
    # # get minimum cost action
    best = costs.index(min(costs))
    best_options[best] += 1
    if uplink_rate > 8000000 and server_cap < 0.5:
        action = 1
    else:
        action = np.random.choice([0, 2])

    # print(action)
    mobile_exec += m_time
    mobile_energy += m_energy
    if m_time > dt:
        mobile_miss_dt += 1

    edge_exec += e_time
    edge_energy += e_energy
    edge_prices += e_price
    if e_time > dt:
        #print(e_time)
        edge_miss_dt += 1

    cloud_exec += c_time
    cloud_energy += c_energy
    cloud_prices += c_price
    if c_time > dt:
        #print(c_time)
        cloud_miss_dt += 1

    if action == 0:
        naive_exec += m_time
        naive_energy += m_energy
        if m_time > dt:
            naive_miss_dt += 1
    elif action == 1:
        naive_exec += e_time
        naive_energy += e_energy
        naive_prices += e_price
        if e_time > dt:
            naive_miss_dt += 1
    else:
        naive_exec += c_time
        naive_energy += c_energy
        naive_prices += c_price
        if c_time > dt:
            naive_miss_dt += 1
    # print(costs)
    return


def update(state_gen):
    # state = env.reset()
    total_step = 1
    for t in itertools.count():
        state = next(state_gen)
        calculate_costs(state)
        total_step += 1

        if total_step > 100000:
            break

    print("complete")
    # print(stats)


if __name__ == '__main__':
    num_of_episodes = 200

    env = Environment()
    state_gen = read_state_from_file()

    mobile_exec = 0
    mobile_energy = 0
    mobile_miss_dt = 0
    edge_exec = 0
    edge_energy = 0
    edge_miss_dt = 0
    edge_prices = 0
    cloud_exec = 0
    cloud_energy = 0
    cloud_miss_dt = 0
    cloud_prices = 0
    naive_exec = 0
    naive_energy = 0
    naive_miss_dt = 0
    naive_prices = 0
    best_options = {
        0: 0,
        1: 0,
        2: 0
    }

    update(state_gen)

    print("mobile-time: ", mobile_energy)
    print("mobile-energy: ", mobile_exec)
    print("mobile-deadline: ", mobile_miss_dt)

    print("edge-time: ", edge_exec)
    print("edge-energy: ", edge_energy)
    print("edge-deadline: ", edge_miss_dt)
    print("edge-price: ", edge_prices)

    print("cloud-time: ", cloud_exec)
    print("cloud-energy: ", cloud_energy)
    print("cloud-deadline: ", cloud_miss_dt)
    print("cloud-prices: ", cloud_prices)

    print("naive-time: ", naive_exec)
    print("naive-energy: ", naive_energy)
    print("naive-deadline: ", naive_miss_dt)
    print("naive-price: ", naive_prices)

    print(best_options)
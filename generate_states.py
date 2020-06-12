import csv
import itertools

from task import get_random_task
from system_parameters import parameter
import random
import numpy as np
import itertools


def cal_uplink_rate():
    # generate a uplink rate in 7-10 Mbps range
    rate = np.random.randint(7, 12)
    rate *= 1000000  # in bit per sec
    return rate


def get_mobile_availability():
    # generate mobile availability in percentage 50-100%
    availability = [0.2, 0.4, 0.6, 0.8, 1.0]
    return np.random.choice(availability)
    # return random.uniform(0.5, 1)


def get_server_availability():
    # generate edge server availability in 0-100%
    availability = [0.2, 0.4, 0.6, 0.8, 1.0]
    return np.random.choice(availability)
    # return random.uniform(0, 1)


def get_energy_availability():
    energies = [342, 300, 250, 200, 150, 100, 50]
    return np.random.choice(energies)


def get_initial_state():
    uplink_rate = cal_uplink_rate()
    mobile_cap = get_mobile_availability()
    server_cap = get_server_availability()
    energy = parameter["total_energy"]  # initial energy level wh, 3110 mAh battery with 110 V voltage
    task = get_random_task()
    state = [task['data'], task['cpu_cycle'], task['dt'], uplink_rate, mobile_cap, server_cap, energy]
    return state


def get_next_state():
    pass


def energy_gen():
    energy = 100
    while True:
        yield energy
        energy -= 10
        if energy < 10:
            energy = 100


def generate_state_trace():
    states = []
    gen = energy_gen()
    final_i = 0
    for i in itertools.count():
        data = random.randint(4096000, 16384000)  # 500KB - 2000 KB
        cpu_cycle = random.randint(1000e6, 10000e6)  # 1000-10000 Mega Cycles
        delay = 4  # 4s in each application
        uplink_rate = cal_uplink_rate()
        mobile_cap = get_mobile_availability()
        row = [data, cpu_cycle, delay, uplink_rate, mobile_cap, next(gen)]
        if row in states:
            continue
        else:
            states.append(row)
        if len(states) >= 150000:
            break
        final_i = i
    print(final_i)
    with open('data/state_trace.csv', 'w+') as file:
        writer = csv.writer(file)
        for num, item in enumerate(states):
            item.insert(0, num + 1)
            writer.writerow(item)


def edge_generator():
    file = 'data/edge_trace.csv'
    for row in open(file):
        yield row


def state_generator():
    file = 'data/state_trace.csv'
    for row in open(file):
        yield row


def read_state_from_file():
    state_gen = state_generator()
    edge_gen = edge_generator()
    while True:
        state = next(state_gen)
        edge_trace = next(edge_gen)
        state = state.split(',')
        edge_trace = edge_trace.split(',')
        state = [float(item) for item in state]
        edge_trace = [float(item) for item in edge_trace]
        state.append(edge_trace[1])
        final_state = [state[1], state[2], state[3] + 6.0, state[4], state[5], state[7], state[6]]
        yield final_state


if __name__ == '__main__':
    # result = read_state_from_file()
    # for i in range(10):
    #     print(next(result))
    generate_state_trace()

import csv
import itertools
import random

import numpy as np

from system_parameters import parameter

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


def get_initial_state(state_generator, battery_obj):
    new_state = next(state_generator)
    battery_level = battery_obj.set_battery_level(new_state[0])
    new_state.append(battery_level)
    return new_state


def get_next_state(state_gen, battery_obj):
    new_state = next(state_gen)
    battery_level = battery_obj.get_battery_level(new_state[0])
    new_state.append(battery_level)
    return new_state


def generate_state_trace():
    edge_trace = {}
    with open('data/edge_trace.csv') as f:
        for line in f:
            (key, val) = line.split(",")
            edge_trace[int(key)] = float(val)

    states = []
    final_i = 1
    minutes = 0
    for i in itertools.count():
        minutes += random.expovariate(1 / 5.0)
        data = random.randint(8388608, 33554432)  # 1MB  - 4MB
        cpu_cycle = random.randint(10000e6, 30000e6)  # 10k-30k Mega Cycles
        delay = 30  # 10s in each application
        uplink_rate = cal_uplink_rate()
        mobile_cap = get_mobile_availability()
        edge_cap = edge_trace[int(minutes / 60)]
        row = [minutes, data, cpu_cycle, uplink_rate, mobile_cap, edge_cap]
        if row in states:
            print("skipping duplicates")
            continue
        else:
            states.append(row)
        if len(states) >= 150000:
            break
        final_i = i
    print(final_i)
    with open('data/state_trace.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for num, item in enumerate(states):
            # item.insert(0, num + 1)
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
    while True:
        state = next(state_gen)
        state = state.split(',')
        state = [float(item) for item in state]
        final_state = [state[0], state[1], state[2], state[3], state[4], state[5]]
        yield final_state


def poisson_job_arrival():
    minutes = 0
    for i in range(100000):
        minutes += random.expovariate(1 / 5.0)
        # print(int(minutes / 60))
        print(minutes / 60)


if __name__ == '__main__':
    # result = read_state_from_file()
    # for i in range(10):
    #     print(next(result))

    generate_state_trace()
    # poisson_job_arrival()

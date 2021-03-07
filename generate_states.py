import csv
import itertools
import math
import random

import numpy as np

from system_parameters import parameter


def cal_uplink_rate():
    # generate a uplink rate in 7-12 Mbps range
    rate = np.random.randint(7, 12)
    rate *= 1000000  # in bit per sec
    return rate


def get_mobile_availability():
    # generate mobile availability in percentage 50-100%
    availability = [0.2, 0.4, 0.6, 0.8, 1.0]
    return np.random.choice(availability)
    # return random.uniform(0.5, 1)


def generate_state_trace():
    edge_trace = {}
    with open('data/edge_trace.csv') as f:
        for line in f:
            (key, val) = line.split(",")
            edge_trace[int(key)] = float(val)

    states = []
    final_i = 1
    hours = 0
    for i in itertools.count():
        # minutes += random.expovariate(12 / 60.0) # 12 job every 60 minutes.
        hours += random.expovariate(15 / 24.0)  # 12 jobs every 24 hours
        cpu_cycle = random.randint(500e6, 2000e6)
        data_size = random.randint(4096000, 8388608)  # 500 kb-1 mb
        uplink_rate = random.randint(8, 13) * 1000000
        # uplink_rate = cal_uplink_rate()
        mobile_cap = get_mobile_availability()
        # edge_cap = edge_trace[int(minutes / 60.0)]
        #edge_cap = edge_trace[math.floor(hours)]
        edge_cap = edge_trace[i+1]
        # print(i, math.floor(hours), edge_cap)
        row = [hours, data_size, cpu_cycle, uplink_rate, mobile_cap, edge_cap]
        if row in states:
            print("skipping duplicates")
            continue
        else:
            states.append(row)
        if len(states) > 120000:
            break
        final_i = i
    print(final_i)
    with open('data/temp_state_trace.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for num, item in enumerate(states):
            # item.insert(0, num + 1)
            writer.writerow(item)


def generate_alternate_trace():
    edge_trace = {}
    with open('data/lyft_edge_trace.csv') as f:
        for line in f:
            (key, val) = line.split(",")
            edge_trace[int(key)] = float(val)

    states = []
    with open('data/new_state_trace.csv', newline='\n') as f:
        for line in f:
            state = line.split(',')
            state = [float(item) for item in state]
            states.append(state)

    alternate_trace = []
    for state in states:
        edge_cap = edge_trace[int(float(state[0]) / 60.0)]
        state[5] = edge_cap
        # cpu_cycle = random.randint(500e6, 2000e6)
        # data_size = random.randint(4096000, 8388608)  # 500 kb-1 mb
        # uplink_rate = random.randint(8, 13) * 1000000
        # state[1] = data_size
        # state[2] = cpu_cycle
        # state[3] = uplink_rate
        # state[4] = get_mobile_availability()
        # row = [state[0], state[1], state[2], state[3], state[4], edge_cap]
        if state in alternate_trace:
            print("skipping duplicates")
            continue
        else:
            alternate_trace.append(state)
        if len(alternate_trace) >= 120000:
            break
        # final_i = i
    with open('data/alternate_state_trace.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for num, item in enumerate(alternate_trace):
            # item.insert(0, num + 1)
            writer.writerow(item)


def edge_generator():
    file = 'data/edge_trace.csv'
    for row in open(file):
        yield row


def state_generator():
    # file = 'data/state_trace.csv'
    file = 'data/temp_state_trace.csv'
    for row in open(file):
        yield row


def read_state_from_file():
    state_gen = state_generator()
    while True:
        state = next(state_gen)
        state = state.split(',')
        state = [float(item) for item in state]
        # 0, 0.1, 0.18, 0.25, 0.32
        # state[5] += 0.1
        # if state[5] >= 1:
        #     state[5] = 0.9999
        # if state[5] < 0:
        #     state[5] = 0
        final_state = [state[0], state[1], state[2], state[3], state[4], state[5]]
        yield final_state


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


def poisson_job_arrival():
    hours = 0
    data = []
    for i in range(1, 120000):
        #minutes += random.expovariate(1 / 5.0)
        hours += random.expovariate(15/24.0)
        # print(int(minutes / 60))
        data.append([hours])
        # print(hours)

    with open("data/job_arrival_times.csv", 'w+', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            #print(item)
            writer.writerow(item)


if __name__ == '__main__':
    # result = read_state_from_file()
    # for i in range(10):
    #     print(next(result))

    generate_state_trace()
    # generate_alternate_trace()
    #poisson_job_arrival()

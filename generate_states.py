from task import get_random_task
from system_parameters import parameter
import random
import numpy as np


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

#
# def get_state_variables():
#     uplink_rate = cal_uplink_rate()  # vary
#     mobile_cap = get_mobile_availability()  # vary based on load
#     server_cap = get_server_availability()  # vary based on load
#     energy = 342  # vary based on load
#     state = [create_task(), uplink_rate, mobile_cap, server_cap, energy]
#     return state

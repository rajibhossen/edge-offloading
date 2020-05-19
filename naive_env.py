import copy
from math import ceil

import gym

from cloud import Cloud
from edge import Edge
from generate_states import *
from mobile import Mobile


class NaiveEnvironment(gym.Env):
    def __init__(self):
        super(NaiveEnvironment, self).__init__()
        self.action_space = ['l', 'e', 'c']
        self.n_actions = len(self.action_space)
        self.state = np.zeros(7)
        self.weight = 0.5
        self.total_energy = parameter["total_energy"]
        self.threshold_energy = 10

    def reset(self):
        self.state = get_initial_state()
        return copy.copy(self.state)

    def step(self, action):
        data = self.state[0]
        cpu_cycle = self.state[1]
        dt = self.state[2]
        uplink_rate = self.state[3]
        mobile_cap = self.state[4]
        server_cap = self.state[5]
        energy_left = self.state[6]

        if action == 0:  # local computing
            device = Mobile(mobile_cap)
            computing_cost, execution_delay, energy_used = device.calculate_cost_naive(cpu_cycle, self.weight)

        elif action == 1:
            edge = Edge(uplink_rate, server_cap)
            computing_cost, execution_delay, energy_used = edge.cal_total_cost_naive(data, cpu_cycle, self.weight)

        else:
            cloud = Cloud(uplink_rate)
            computing_cost, execution_delay, energy_used = cloud.cal_total_cost_naive(data, cpu_cycle, self.weight)

        energy_used = ceil(energy_used)
        energy_left = energy_left - energy_used

        if energy_left < self.threshold_energy:
            reward = -computing_cost
            done = True
            self.state[0] = -1
            self.state[1] = -1
            self.state[2] = -1
            self.state[3] = -1
            self.state[4] = -1
            self.state[5] = -1
            self.state[6] = -1
        else:
            done = False
            # print(execution_delay, dt)
            if execution_delay > dt:
                reward = parameter['max_penalty']
                # done = False
                task = get_random_task()
                self.state[0] = task['data']
                self.state[1] = task['cpu_cycle']
                self.state[2] = task['dt']
                self.state[3] = cal_uplink_rate()
                self.state[4] = get_mobile_availability()
                self.state[5] = get_server_availability()
                self.state[6] = energy_left
                # self.state[6] = get_energy_availability()
            else:
                # compare with mobile computation and
                # positive or negative based on local computing
                reward = -computing_cost
                # done = False
                task = get_random_task()
                self.state[0] = task['data']
                self.state[1] = task['cpu_cycle']
                self.state[2] = task['dt']
                self.state[3] = cal_uplink_rate()
                self.state[4] = get_mobile_availability()
                self.state[5] = get_server_availability()
                self.state[6] = energy_left
                # self.state[6] = get_energy_availability()

        return self.state, reward, done

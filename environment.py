import copy
import csv
from math import ceil

import gym
import numpy as np
from cloud import Cloud
from edge import Edge
from generate_states import read_state_from_file
from mobile import Mobile
from system_parameters import parameter


class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['l', 'e', 'c']
        self.n_actions = len(self.action_space)
        self.state = np.zeros(7)
        self.weight = 0.8
        self.total_energy = parameter["total_energy"]
        self.threshold_energy = 10
        self.get_state = read_state_from_file()
        self.missed_deadline = 0

    def render(self, mode='human'):
        pass

    def reset(self):
        self.state = next(self.get_state)
        # self.state = get_initial_state()
        return copy.copy(self.state)

    def step(self, action):
        data = self.state[0]
        cpu_cycle = self.state[1]
        dt = self.state[2]
        uplink_rate = self.state[3]
        mobile_cap = self.state[4]
        server_cap = self.state[5]
        energy_assign = self.state[6]

        device = Mobile(mobile_cap)
        energy_factor = energy_assign / self.total_energy
        energy_factor = 1 - energy_factor
        m_total, m_time, m_energy = device.calculate_total_cost(cpu_cycle, self.weight, energy_factor)
        if action == 0:  # local computing
            # computing_cost, execution_delay, energy_used = device.calculate_total_cost(task, self.weight,
            # energy_factor)
            computing_cost, execution_delay, energy_used = m_total, m_time, m_energy
        elif action == 1:  # offload to edge
            edge = Edge(uplink_rate, server_cap)
            computing_cost, execution_delay, energy_used = edge.cal_total_cost(data, cpu_cycle, self.weight,
                                                                               energy_factor)
        else:
            cloud = Cloud(uplink_rate)
            computing_cost, execution_delay, energy_used = cloud.cal_total_cost(data, cpu_cycle, self.weight,
                                                                                energy_factor)

        # energy_used = ceil(energy_used)
        energy_left = energy_assign - energy_used
        print(action, computing_cost, execution_delay, energy_used)
        # scaled_reward = ((m_total - computing_cost) / m_total) * 100.0
        if energy_used > energy_assign:
            reward = parameter['max_penalty']
            done = True
            self.state = [-1 for i in range(6)]
            return self.state, reward, done

        if energy_left < self.threshold_energy:
            reward = -computing_cost
            done = True
            self.state = [-1 for i in range(6)]
        else:
            done = False
            # print(execution_delay, dt)
            if execution_delay > dt:
                self.missed_deadline += 1
                reward = parameter['max_penalty']
                self.state = next(self.get_state)
            else:
                reward = -computing_cost
                self.state = next(self.get_state)

        return self.state, reward, done


if __name__ == '__main__':
    env = Environment()
    ini = env.reset()
    print(ini)
    print(env.step(0))
    # print(env.read_states())

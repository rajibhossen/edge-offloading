import copy
import csv
from math import ceil

import gym
import numpy as np
from cloud import Cloud
from edge import Edge
from generate_states import read_state_from_file, get_initial_state, get_next_state
from mobile import Mobile
from system_parameters import parameter
from BatteryEnergy import Energy


class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['l', 'e', 'c']
        self.n_actions = len(self.action_space)
        self.state = np.zeros(7)
        self.battery = Energy()
        self.threshold_energy = 50
        self.get_state = read_state_from_file()
        self.exe_delay = 0
        self.tot_energy_cost = 0
        self.tot_off_cost = 0

    def render(self, mode='human'):
        pass

    def reset(self):
        # self.state = next(self.get_state)
        self.state = get_initial_state(self.get_state, self.battery)
        return copy.copy(self.state)

    def step(self, action):
        timeid = self.state[0]
        data = self.state[1]
        cpu_cycle = self.state[2]
        uplink_rate = self.state[3]
        mobile_cap = self.state[4]
        server_cap = self.state[5]
        energy_assign = self.state[6]

        device = Mobile(mobile_cap)
        m_total, m_time, m_energy = device.calculate_total_cost(cpu_cycle)
        if action == 0:  # local computing
            computing_cost, execution_delay, energy_used, off_price = m_total, m_time, m_energy, 0
        elif action == 1:  # offload to edge
            edge = Edge(uplink_rate, server_cap)
            computing_cost, execution_delay, energy_used, off_price = edge.cal_total_cost(data, cpu_cycle)
        else:
            cloud = Cloud(uplink_rate)
            computing_cost, execution_delay, energy_used, off_price = cloud.cal_total_cost(data, cpu_cycle)

        self.exe_delay += execution_delay
        self.tot_energy_cost += energy_used
        self.tot_off_cost += off_price
        energy_left = energy_assign - energy_used
        self.battery.update_energy(energy_used)
        # print(action, computing_cost, execution_delay, energy_used)
        # scaled_reward = ((m_total - computing_cost) / m_total) * 100.0

        # new reward system
        # done = False
        # # if previous cost is greater than current cost, -1 reward. initially 0 reward
        # if self.previous_cost < computing_cost and self.previous_cost != 0:
        #     reward = 1
        # elif self.previous_cost > computing_cost and self.previous_cost != 0:
        #     reward = -1
        # else:
        #     reward = 0
        #
        # if energy_used > energy_assign:
        #     reward = -10
        # if execution_delay > dt:
        #     reward = -10
        #     self.missed_deadline += 1
        #
        # # episode terminal condition
        # if energy_left < self.threshold_energy:
        #     done = True
        #     self.state = [-1 for i in range(7)]
        # else:
        #     self.state = next(self.get_state)
        #
        # self.previous_cost = computing_cost

        # previous reward system
        done = False

        # scaled_reward = (m_total - computing_cost) / m_total * 1.0

        if energy_left < self.threshold_energy:
            # reward = -computing_cost
            reward = parameter['max_penalty']
            done = True
            self.state = [-1 for i in range(7)]
            return self.state, reward, done

        reward = -computing_cost
        self.state = get_next_state(self.get_state, self.battery)
        return self.state, reward, done


if __name__ == '__main__':
    env = Environment()
    ini = env.reset()
    print(ini)
    print(env.step(0))
    # print(env.read_states())

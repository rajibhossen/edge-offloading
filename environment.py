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
        self.action_space = ['l', 'e']
        self.n_actions = len(self.action_space)
        self.state = np.zeros(7)
        self.battery = Energy()
        self.threshold_energy = 20
        # self.get_state = None
        self.get_state = read_state_from_file()
        self.exe_delay = []
        self.trans_delay = []
        self.proc_energy = []
        self.trans_energy = []
        self.tot_off_cost = []
        self.total_cost = []
        self.off_decisions = {0: 0, 1: 0}
        self.off_from_edge = 0
        self.state_counter = 0
        self.total_recharge = 0

    def render(self, mode='human'):
        pass

    def reset(self):
        # self.get_state = read_state_from_file()
        # self.state_counter = 0
        self.total_recharge += 1
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
        self.off_decisions[action] += 1

        device = Mobile(mobile_cap)
        m_total, m_time, m_energy = device.calculate_total_cost(cpu_cycle)
        off_edge = 0
        transmission_delay = 0
        proc_energy = 0
        trans_energy = 0
        if action == 0:  # local computing
            computing_cost, execution_delay, energy_used, off_price = m_total, m_time, m_energy, 0
            proc_energy = energy_used
        elif action == 1:  # offload to edge
            edge = Edge(uplink_rate, server_cap)
            computing_cost, transmission_delay, execution_delay, energy_used, off_price, off_edge = edge.cal_total_cost(data, cpu_cycle)
            trans_energy = energy_used
        else:
            cloud = Cloud(uplink_rate)
            computing_cost, transmission_delay, execution_delay, energy_used, off_price = cloud.cal_total_cost(data, cpu_cycle)
            trans_energy = energy_used

        # self.total_cost += computing_cost
        # self.exe_delay += execution_delay
        # self.trans_delay += transmission_delay
        # self.proc_energy += proc_energy
        # self.trans_energy += trans_energy
        # self.tot_off_cost += off_price
        # self.off_from_edge += off_edge
        self.total_cost.append([computing_cost])
        self.exe_delay.append([execution_delay+transmission_delay])
        #self.trans_delay.append([transmission_delay])
        self.proc_energy.append([proc_energy+trans_energy])
        #self.trans_energy.append([trans_energy])
        self.off_from_edge += off_edge

        energy_left = energy_assign - energy_used
        self.battery.update_energy(energy_used)

        # print(action, computing_cost, execution_delay, energy_used)
        # scaled_reward = ((m_total - computing_cost) / m_total) * 100.0

        # previous reward system
        done = False
        # scaled_reward = (m_total - computing_cost) / m_total * 1.0
        # # print("Total: %f (t-%f, e-%f, m-%f)" % (computing_cost, execution_delay, energy_used, off_price))

        if energy_left < self.threshold_energy:
            # reward = -computing_cost
            reward = parameter['max_penalty']
            done = True
            self.state = [-1 for i in range(7)]
            return self.state, reward, done

        reward = -computing_cost
        self.state = get_next_state(self.get_state, self.battery)
        return self.state, reward, done

        # self.state_counter += 1
        # if energy_left < self.threshold_energy:
        #     reward = parameter['max_penalty']
        #     self.battery.set_battery_level(timeid)
        # else:
        #     reward = -computing_cost
        #
        # if self.state_counter >= 10000:
        #     done = True
        #     self.state = [-1 for i in range(7)]
        #     return self.state, reward, done
        # else:
        #     self.state = get_next_state(self.get_state, self.battery)
        #     return self.state, reward, done


if __name__ == '__main__':
    env = Environment()
    ini = env.reset()
    print(ini)
    print(env.step(0))
    # print(env.read_states())

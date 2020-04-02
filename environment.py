import copy
import logging

import gym

from cloud import Cloud
from edge import Edge
from generate_states import *
from mobile import Mobile


class Environment(gym.Env):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['l', 'e', 'c']
        self.n_actions = len(self.action_space)
        self.state = np.zeros(5)
        self.weight = 0.5
        self.cloud_cap = parameter["cloud_cap"]
        self.total_energy = parameter["total_energy"]
        logging.basicConfig(filename="data/offloading.log", level=logging.DEBUG)

    def reset(self):
        self.state = get_initial_state()
        return copy.copy(self.state)

    def step(self, action):
        task = self.state[0]
        uplink_rate = self.state[1]
        mobile_cap = self.state[2]
        server_cap = self.state[3]
        self.energy_left = self.state[4]

        device = Mobile(mobile_cap)
        energy_factor = self.energy_left / self.total_energy
        energy_factor = 1 - energy_factor
        m_total, m_time, m_energy = device.calculate_total_cost(task, self.weight, energy_factor)
        if action == 0:  # local computing
            # computing_cost, execution_delay, energy_used = device.calculate_total_cost(task, self.weight,
            # energy_factor)
            computing_cost, execution_delay, energy_used = m_total, m_time, m_energy
        elif action == 1:  # offload to edge
            edge = Edge(uplink_rate, server_cap)
            computing_cost, execution_delay, energy_used = edge.cal_total_cost(task, self.weight, energy_factor)
        else:
            cloud = Cloud(uplink_rate, self.cloud_cap)
            computing_cost, execution_delay, energy_used = cloud.cal_total_cost(task, self.weight, energy_factor)

        self.energy_left = self.energy_left - energy_used

        # if self.energy_left < 5:
        #     self.state[0] = -1
        #     self.state[1] = -1
        #     self.state[2] = -1
        #     self.state[3] = -1
        #     self.state[4] = -1
        #     done = True
        #     reward = parameter['max_penalty']
        #     return copy.copy(self.state), reward, done

        # logging.info(str(self.state) + ", " + str(action) + ", " + str(computing_cost) + ", " + str(
        #     execution_delay) + ", " + str(energy_used))
        # if self.energy_left < 50:
        #     reward = parameter['max_penalty']
        #     done = True
        #     self.state[0] = -1
        #     self.state[1] = -1
        #     self.state[2] = -1
        #     self.state[3] = -1
        #     self.state[4] = -1
        # else:
        # check battery level and reset

        # if self.energy_left < 50:
        #     done = True
        # else:
        #     done = False
        scaled_reward = ((m_total - computing_cost) / m_total)*100.0
        if self.energy_left < 50:
            reward = scaled_reward
            done = True
            self.state[0] = -1
            self.state[1] = -1
            self.state[2] = -1
            self.state[3] = -1
            self.state[4] = -1
        else:
            done = False
            if execution_delay > task['delay_tolerance']:
                reward = parameter['max_penalty']
                # done = False
                self.state[0] = get_random_task()
                self.state[1] = cal_uplink_rate()
                self.state[2] = get_mobile_availability()
                self.state[3] = get_server_availability()
                # self.state[4] = self.energy_left
                self.state[4] = get_energy_availability()
            else:
                # compare with mobile computation and
                # positive or negative based on local computing
                reward = scaled_reward
                # done = False
                self.state[0] = get_random_task()
                self.state[1] = cal_uplink_rate()
                self.state[2] = get_mobile_availability()
                self.state[3] = get_server_availability()
                # self.state[4] = self.energy_left
                self.state[4] = get_energy_availability()

        return copy.copy(self.state), reward, done

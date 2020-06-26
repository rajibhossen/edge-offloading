from system_parameters import parameter
import task
import random
from cloud import Cloud


class Edge:
    def __init__(self, uplink_rate, utilization):
        self.tr_power = parameter['tr_power']
        self.tail_latency_energy = parameter['tail_energy']
        self.uplink_rate = uplink_rate
        self.execution_cap = parameter["edge_com_cap"] * (1 - utilization)
        self.server_utilization = utilization
        self.w1 = parameter['w1']
        self.w2 = parameter['w2']
        self.w3 = parameter['w3']

    def cal_transmit_time(self, data):
        tr_time = data / self.uplink_rate
        return tr_time

    def cal_transmit_energy(self, data):
        tr_energy = self.tr_power * self.cal_transmit_time(data) + self.tail_latency_energy * 2
        # tr_energy = self.tr_power * self.cal_transmit_time(data)
        return tr_energy

    def cal_processing_time(self, cpu_cycle):
        proc_time = cpu_cycle / self.execution_cap
        # proc_time = proc_time  # in s
        return proc_time

    def cal_price(self, proc_time):
        expense = proc_time * parameter['edge_cps'] + parameter['edge_request']
        return expense

    def cal_total_cost(self, data, cpu_cycle):
        """
        if utilization is less than 50%, process here, otherwise offload to cloud by 2x.
        if utilization = 0.6, 0.6-0.5 = 0.1*2 = 20% chance to offload in cloud
        """
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)

        process_here = self.server_utilization - 0.5

        if random.uniform(0, 1) >= (2 * process_here):
            proc_time = self.cal_processing_time(cpu_cycle)
            money = self.cal_price(proc_time)
            time = edge_tr_time + proc_time
            total = self.w1 * time + self.w2 * energy + self.w3 * money
            return total, time, energy, money
        else:
            cloud = Cloud(self.uplink_rate)
            cloud_tr_time = cloud.cal_transmit_from_edge(data)
            proc_time = cloud.cal_processing_time(cpu_cycle)
            money = cloud.cal_price(proc_time)
            time = edge_tr_time + cloud_tr_time + proc_time
            total = self.w1 * time + self.w2 * energy + money * self.w3
            return total, time, energy, money

    def cal_total_cost_naive(self, data, cpu_cycle):
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)
        proc_time = self.cal_processing_time(cpu_cycle)
        money = self.cal_price(proc_time)
        time = edge_tr_time + proc_time
        total = self.w1 * time + self.w2 * energy + money * self.w3
        # total = time + money + energy
        return total, time, energy, money


if __name__ == '__main__':
    for cap in [0.2, 0.4, 0.6, 0.8, 1]:
        edge = Edge(7000000, cap)
        job = task.get_fixed_task()
        print(edge.cal_total_cost(job['data'], job['cpu_cycle']))

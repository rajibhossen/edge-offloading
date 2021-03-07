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
        # self.off_from_edge = off_from_edge

    def cal_transmit_time(self, data):
        tr_time = data / self.uplink_rate
        return tr_time

    def cal_transmit_energy(self, data):
        tr_time = self.cal_transmit_time(data)
        tr_energy = self.tr_power * tr_time + self.tail_latency_energy * 1.5
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
        if utilization is less than 50%, process here, otherwise offload to cloud by 2.5x.
        if utilization = 0.6, 0.6-0.5 = 0.1*2.5 = 25% chance to offload in cloud
        for 10.67% = 0.51, 20.87 = 0.4, 29.78=0.33, 40.26 = 0.26, 50 = 0.19
        """
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)

        process_here = self.server_utilization - 0.4
        # process_here = 1

        if random.uniform(0, 1) >= 2.5 * process_here:
            proc_time = self.cal_processing_time(cpu_cycle)
            money = self.cal_price(proc_time)
            time = edge_tr_time + proc_time
            total = self.w1 * time + self.w2 * energy + self.w3 * money
            # return total, time, energy, money, 0
            return total, edge_tr_time, proc_time, energy, money, 0
        else:
            cloud = Cloud(self.uplink_rate)
            cloud_tr_time = cloud.cal_transmit_from_edge(data)
            proc_time = cloud.cal_processing_time(cpu_cycle)
            money = cloud.cal_price(proc_time)
            time = edge_tr_time + cloud_tr_time + proc_time
            total = self.w1 * time + self.w2 * energy + money * self.w3
            # self.off_from_edge += 1
            # return total, time, energy, money, 1
            return total, edge_tr_time + cloud_tr_time, proc_time, energy, money, 1

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

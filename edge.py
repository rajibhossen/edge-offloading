from system_parameters import parameter
import task
import random
from cloud import Cloud


class Edge:
    def __init__(self, uplink_rate, utilization):
        self.tr_power = parameter['tr_power']
        self.tail_latency_energy = parameter['tail_energy']
        self.tail_duration = parameter["tail_duration"]
        self.uplink_rate = uplink_rate
        self.execution_cap = parameter["edge_com_cap"] * utilization
        self.server_utilization = utilization
        self.price = parameter["cph_edge"]

    def cal_transmit_time(self, data):
        tr_time = data / self.uplink_rate
        return tr_time

    def cal_transmit_energy(self, data):
        tr_energy = self.tr_power * self.cal_transmit_time(data) + self.tail_latency_energy * self.tail_duration
        # tr_energy = self.tr_power * self.cal_transmit_time(data)
        return tr_energy

    def cal_processing_time(self, cpu_cycle):
        proc_time = cpu_cycle / self.execution_cap
        # proc_time = proc_time  # in s
        return proc_time

    def cal_price(self, proc_time):
        expense = proc_time * self.price
        return expense

    def cal_total_cost(self, data, cpu_cycle, weight, energy_factor):
        """
        if utilization is less than 50%, process here, otherwise offload to cloud by 2x.
        if utilization = 0.6, 0.6-0.5 = 0.1*2 = 20% chance to offload in cloud
        :param data:
        :param cpu_cycle:
        :param weight:
        :param energy_factor:
        :return:
        """
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)

        process_here = self.server_utilization - 0.5

        if random.uniform(0, 1) < (2 * process_here):
            proc_time = self.cal_processing_time(cpu_cycle)
            money = self.cal_price(proc_time)
            time = edge_tr_time + proc_time
            energy_impact = energy + energy * energy_factor
            total = (1 - weight) * time + weight * energy_impact + money
            return total, time, energy
        else:
            cloud = Cloud(self.uplink_rate)
            cloud_tr_time = cloud.cal_transmit_from_edge(data)
            proc_time = cloud.cal_processing_time(cpu_cycle)
            money = cloud.cal_price(proc_time)
            time = edge_tr_time + cloud_tr_time + proc_time
            energy_impact = energy + energy * energy_factor
            total = (1 - weight) * time + weight * energy_impact + money
            return total, time, energy

    def cal_total_cost_naive(self, data, cpu_cycle, weight):
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)
        proc_time = self.cal_processing_time(cpu_cycle)
        money = self.cal_price(proc_time)
        time = edge_tr_time + proc_time
        total = (1 - weight) * time + weight * energy + money
        return total, time, energy


if __name__ == '__main__':
    edge = Edge(7000000, 0.6)
    for u in range(7, 12):
        u_rate = u * 1000000
        for cap in [0.2, 0.4, 0.6, 0.8, 1]:
            edge = Edge(u_rate, cap)
            job = task.get_fixed_task()
            print(edge.cal_total_cost(job['data'], job['cpu_cycle'], 0.5, 0))

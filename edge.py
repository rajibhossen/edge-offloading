from system_parameters import parameter
import task
import random
from cloud import Cloud


class Edge:
    def __init__(self, uplink_rate, probability):
        self.tr_power = parameter['tr_power']
        self.tail_latency_energy = parameter['tail_energy']
        self.tail_duration = parameter["tail_duration"]
        self.uplink_rate = uplink_rate
        self.execution_cap = parameter["edge_com_cap"]
        self.process_here = probability
        self.price = parameter["cph_edge"]

    def cal_transmit_time(self, data):
        tr_time = data / self.uplink_rate
        return tr_time

    def cal_transmit_energy(self, data):
        tr_energy = self.tr_power * self.cal_transmit_time(data) + self.tail_latency_energy * self.tail_duration
        return tr_energy

    def cal_processing_time(self, cpu_cycle):
        proc_time = cpu_cycle / self.execution_cap
        proc_time = proc_time  # in s
        return proc_time

    def cal_price(self, proc_time):
        expense = proc_time * self.price
        return expense

    def cal_total_cost(self, data, cpu_cycle, weight, energy_factor):
        edge_tr_time = self.cal_transmit_time(data)
        energy = self.cal_transmit_energy(data)

        if random.uniform(0, 1) < self.process_here:
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


if __name__ == '__main__':
    edge = Edge(7000000, 0.6)
    # task = get_fixed_task()
    for app in task.applications:
        job = task.make_task_from_applications(app)
        print(job['cpu_cycle']/(8*1024))
        print(edge.cal_total_cost(job['data'], job['cpu_cycle'], 0.5, 0))
    # print("transmit energy: ", edge.cal_transmit_energy(task['data']))
    # print("transmit time: ", edge.cal_transmit_time(task['data']))
    # # print(edge.cal_transmit_time(task) + edge.cal_processing_time(task))
    # print(edge.cal_total_cost(task['data'], task['cpu_cycle'], 0.5, 0))

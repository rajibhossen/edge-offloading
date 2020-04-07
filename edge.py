from system_parameters import parameter
from task import get_fixed_task


class Edge:
    def __init__(self, uplink_rate, execution_cap):
        self.tr_power = parameter['tr_power']
        self.tail_latency_energy = parameter['tail_energy']
        self.tail_duration = parameter["tail_duration"]
        self.uplink_rate = uplink_rate
        self.execution_cap = execution_cap * parameter["edge_com_cap"]
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
        tr_time = self.cal_transmit_time(data)
        proc_time = self.cal_processing_time(cpu_cycle)
        energy = self.cal_transmit_energy(data)
        money = self.cal_price(proc_time)
        time = tr_time + proc_time
        energy_impact = energy + energy * energy_factor
        total = (1 - weight) * time + weight * energy_impact + money
        return total, time, energy


if __name__ == '__main__':
    edge = Edge(7000000, 1)
    task = get_fixed_task()
    print("transmit energy: ", edge.cal_transmit_energy(task['data']))
    print("transmit time: ", edge.cal_transmit_time(task['data']))
    # print(edge.cal_transmit_time(task) + edge.cal_processing_time(task))
    print(edge.cal_total_cost(task['data'], task['cpu_cycle'], 0.5, 0))

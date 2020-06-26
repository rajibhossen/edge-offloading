from system_parameters import parameter
import numpy as np
import task


# np.random.seed(1)


class Cloud:
    def __init__(self, uplink_rate):
        self.tr_power = parameter['tr_power']
        self.tail_latency_energy = parameter['tail_energy']
        self.tail_duration = parameter["tail_duration"]
        self.uplink_rate = uplink_rate
        self.execution_cap = parameter['cloud_com_cap'] * parameter["cloud_cap"]
        self.w1 = parameter['w1']
        self.w2 = parameter['w2']
        self.w3 = parameter['w3']

    def cal_propagation_delay(self):
        # delay = self.distance / self.propagation_speed
        # delay = 0.25 # amazon us-east-1 average latency for a day. 250ms,
        # using normal distribution, mean = 250ms, std=448ms
        while True:
            delay = np.random.randint(300, 500)
            if delay < 0:
                continue
            else:
                delay = delay / 1000.0
                break
        return delay

    def cal_transmit_time(self, data):
        tr_time = data / self.uplink_rate
        tr_time += tr_time  # from mobile to edge and then edge to cloud. so, transmit time is twice.
        tr_time += self.cal_propagation_delay()
        tr_time += np.random.randint(5, 10)  # queuing delay
        return tr_time

    def cal_transmit_from_edge(self, data):
        tr_time = data / self.uplink_rate
        tr_time += self.cal_propagation_delay()
        # tr_time += np.random.randint(3, 4)  # queuing delay
        return tr_time

    def cal_transmit_energy(self, data):
        tr_time = data / self.uplink_rate
        tr_energy = self.tr_power * tr_time + self.tail_latency_energy * 3
        return tr_energy

    def cal_processing_time(self, cpu_cycle):
        proc_time = cpu_cycle / self.execution_cap
        return proc_time

    def cal_price(self, proc_time):
        expense = proc_time * parameter['cloud_cps'] + parameter['cloud_request']
        return expense

    def cal_total_cost(self, data, cpu_cycle):
        tr_time = self.cal_transmit_time(data)
        proc_time = self.cal_processing_time(cpu_cycle)
        energy = self.cal_transmit_energy(data)
        money = self.cal_price(proc_time)
        time = tr_time + proc_time
        total = self.w1 * time + self.w2 * energy + self.w3 * money
        return total, time, energy, money

    def cal_total_cost_naive(self, data, cpu_cycle):
        tr_time = self.cal_transmit_time(data)
        proc_time = self.cal_processing_time(cpu_cycle)
        energy = self.cal_transmit_energy(data)
        money = self.cal_price(proc_time)
        time = tr_time + proc_time
        total = self.w1 * time + self.w2 * energy + money * self.w3
        # total = time + money + energy
        return total, time, energy, money


if __name__ == '__main__':
    cloud = Cloud(7000000)
    t = task.get_fixed_task()
    # for app in task.applications:
    job = task.get_fixed_task()
    print(cloud.cal_total_cost(job['data'], job['cpu_cycle']))
    # print(cloud.cal_transmit_energy(task))
    # print(cloud.cal_processing_time(task['data']))
    # print(cloud.cal_transmit_time(task) + cloud.cal_processing_time(task))
    # print(cloud.cal_total_cost(task['data'], task['cpu_cycle'], 0.5, 0))

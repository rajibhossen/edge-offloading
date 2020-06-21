from system_parameters import parameter
import task


class Mobile:
    def __init__(self, mobile_cap):
        self.esc = parameter["esc"]  # effective switched capacitance
        self.computing_capability = parameter['mobile_com_cap']
        self.limited_capability = parameter['mobile_com_cap'] * mobile_cap
        self.w1 = parameter['w1']

    def calculate_time(self, cpu_cycle):
        time = cpu_cycle / self.limited_capability
        return time  # in s

    def calculate_energy(self, cpu_cycle):
        energy = self.esc * (self.computing_capability ** 2) * cpu_cycle
        return energy

    def calculate_total_cost(self, cpu_cycle):
        time = self.calculate_time(cpu_cycle)
        energy = self.calculate_energy(cpu_cycle)
        total = self.w1 * time + energy
        #total = time + energy
        return total, time, energy

    def calculate_cost_naive(self, cpu_cycle):
        time = self.calculate_time(cpu_cycle)
        energy = self.calculate_energy(cpu_cycle)
        total = self.w1 * time + energy
        return total, time, energy


if __name__ == '__main__':
    for cap in [0.2, 0.4, 0.6, 0.8, 1]:
        mobile = Mobile(cap)
        job = task.get_fixed_task()
        print(mobile.calculate_total_cost(job['cpu_cycle']))

    # for app in task.applications:
    #     job = task.make_task_from_applications(app)
    #     print(job['cpu_cycle']/(8*1024))
    #     # print(mobile.cal_energy_by_data(job['data']))
    #     # print("energy: ", mobile.calculate_energy(job['cpu_cycle']))
    #     print(mobile.calculate_total_cost(job['cpu_cycle'], 0.5, 0))
    # # print("processing time: ", mobile.calculate_time(task['cpu_cycle']))
    #

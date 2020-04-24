from system_parameters import parameter
import task


class Mobile:
    def __init__(self, mobile_cap):
        self.esc = parameter["esc"]  # effective switched capacitance
        self.computing_capability = parameter['mobile_com_cap']
        self.limited_capability = parameter['mobile_com_cap'] * mobile_cap

    def calculate_time(self, cpu_cycle):
        time = cpu_cycle / self.limited_capability
        return time  # in s

    def calculate_energy(self, cpu_cycle):
        energy = self.esc * (self.computing_capability ** 2) * cpu_cycle
        return energy

    def cal_energy_by_data(self, data):
        energy = 3.25e-7 * data
        return energy

    def calculate_total_cost(self, cpu_cycle, weight, energy_factor):
        time = self.calculate_time(cpu_cycle)
        energy = self.calculate_energy(cpu_cycle)
        energy_impact = energy + energy * energy_factor
        total = (1 - weight) * time + weight * energy_impact
        return total, time, energy


if __name__ == '__main__':
    for cap in [0.2, 0.4, 0.6, 0.8, 1]:
        mobile = Mobile(cap)
        job = task.get_fixed_task()
        print(mobile.calculate_total_cost(job['cpu_cycle'], 0.5, 0))

    # for app in task.applications:
    #     job = task.make_task_from_applications(app)
    #     print(job['cpu_cycle']/(8*1024))
    #     # print(mobile.cal_energy_by_data(job['data']))
    #     # print("energy: ", mobile.calculate_energy(job['cpu_cycle']))
    #     print(mobile.calculate_total_cost(job['cpu_cycle'], 0.5, 0))
    # # print("processing time: ", mobile.calculate_time(task['cpu_cycle']))
    #

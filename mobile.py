from system_parameters import parameter
import task


class Mobile:
    def __init__(self, mobile_cap):
        self.esc = parameter["esc"]  # effective switched capacitance
        self.computing_capability = parameter['mobile_com_cap'] * mobile_cap

    def calculate_time(self, task):
        time = task['cpu_cycle'] / self.computing_capability
        return time  # in s

    def calculate_energy(self, task):
        energy = self.esc * self.computing_capability ** 2 * task['cpu_cycle']
        return energy

    def calculate_total_cost(self, task, weight, energy_factor):
        time = self.calculate_time(task)
        energy = self.calculate_energy(task)
        energy_impact = energy + energy * energy_factor
        total = (1 - weight) * time + weight * energy_impact
        return total, time, energy


if __name__ == '__main__':
    mobile = Mobile(0.8)

    task = task.get_fixed_task()
    print("energy: ", mobile.calculate_energy(task))
    print("processing time: ", mobile.calculate_time(task))
    print(mobile.calculate_total_cost(task, 0.6, 0))

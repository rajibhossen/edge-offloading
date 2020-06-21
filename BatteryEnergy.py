from system_parameters import parameter


class Energy:
    def __init__(self):
        self.last_recharged = None
        self.available_energy = None

    def set_battery_level(self, timeid):
        self.last_recharged = timeid
        self.available_energy = parameter['total_energy']
        return self.available_energy

    def get_battery_level(self, timeid):
        duration = timeid - self.last_recharged
        decay = duration * 0.11875
        self.available_energy -= decay
        return self.available_energy

    def update_energy(self, energy_used):
        self.available_energy -= energy_used


if __name__ == '__main__':
    energy = Energy()
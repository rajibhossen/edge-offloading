import itertools

import matplotlib.style

from cloud import Cloud
from edge import Edge
from environment import Environment
from generate_states import read_state_from_file
from mobile import Mobile

matplotlib.style.use('ggplot')
import numpy as np


def update(episodes):
    global max_total
    total_step = 1
    for episode in range(episodes):
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (episode, t))

            mobile = Mobile(state[4])
            mc, mt, me = mobile.calculate_total_cost(state[2])

            edge = Edge(state[3], state[5])
            ec, et, ee, ep = edge.cal_total_cost(state[1], state[2])
            cloud = Cloud(state[3])
            cc, ct, ce, cp = cloud.cal_total_cost(state[1], state[2])
            total_costs = [mc, ec, cc]
            bests[total_costs.index(min(total_costs))] += 1

            max_total = max(total_costs) if max(total_costs) > max_total else max_total

            total_time = [mt, et, ct]
            best_time[total_time.index(min(total_time))] += 1
            total_energy = [me, ee, ce]
            best_energy[total_energy.index(min(total_energy))] += 1
            total_m = [ep, cp]
            best_money[total_m.index(min(total_m))] += 1
            # print(total_costs)
            action = 1
            state_, reward, done = env.step(action)

            state = state_
            total_step += 1
            if done:
                break
        if total_step >= 100013:
            break

    print("complete")


if __name__ == '__main__':
    num_of_episodes = 50000

    env = Environment()
    bests = {0: 0, 1: 0, 2: 0}
    best_time = {0: 0, 1: 0, 2: 0}
    best_energy = {0: 0, 1: 0, 2: 0}
    best_money = {0: 0, 1: 0, 2: 0}
    max_total = 0
    update(num_of_episodes)
    print(bests)
    print(best_time)
    print(best_energy)
    print(best_money)
    print("Max: ", max_total)
    print("Total Costs:", env.total_cost)
    print("Total Execution Time: ", env.exe_delay)
    print("Total Energy cost: ", env.tot_energy_cost)
    print("Total Money for offloading: ", env.tot_off_cost)

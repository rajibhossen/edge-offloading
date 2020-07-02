import itertools

import matplotlib.style

from cloud import Cloud
from edge import Edge
from environment import Environment
from generate_states import read_state_from_file
from mobile import Mobile

matplotlib.style.use('ggplot')
import numpy as np


def get_naive_action(state):
    time_id, data, cpu_cycle, uplink_r, mobile_cap, edge_cap, energy = state
    mobile = Mobile(mobile_cap)
    mc, mt, me = mobile.calculate_cost_naive(cpu_cycle)
    edge = Edge(uplink_r, edge_cap,0)
    ec, et, ee, ep = edge.cal_total_cost_naive(data, cpu_cycle)
    cloud = Cloud(uplink_r)
    cc, ct, ce, cp = cloud.cal_total_cost_naive(data, cpu_cycle)

    if energy > 100:  # battery is greater than certain percentage, consider edge, cloud
        action = 1 if ec < cc else 2
    elif uplink_r > 80000 and edge_cap < 0.7:  # good uplink rate and edge cap, consider edge and mobile
        action = 1 if ec < mc else 0
    else:  # all other cases consider mobile and cloud
        action = 0 if mc < cc else 2

    return action


def update(episodes):
    global max_total
    total_step = 1
    for episode in range(episodes):
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (episode, t))

            time_id, data, cpu_cycle, uplink_r, mobile_cap, edge_cap, energy = state

            mobile = Mobile(mobile_cap)
            mc, mt, me = mobile.calculate_total_cost(cpu_cycle)
            edge = Edge(uplink_r, edge_cap, 0)
            ec, et, ee, ep, _ = edge.cal_total_cost(data, cpu_cycle)
            cloud = Cloud(uplink_r)
            cc, ct, ce, cp = cloud.cal_total_cost(data, cpu_cycle)
            total_costs = [mc, ec, cc]

            bests[total_costs.index(min(total_costs))] += 1
            #
            # max_total = max(total_costs) if max(total_costs) > max_total else max_total
            #
            total_time = [mt, et, ct]
            best_time[total_time.index(min(total_time))] += 1
            total_energy = [me, ee, ce]
            best_energy[total_energy.index(min(total_energy))] += 1
            #total_m = [ep, cp]
            #best_money[total_m.index(min(total_m))] += 1
            # print(total_costs)
            #action = total_costs.index(min(total_costs))
            action = get_naive_action(state)

            # action = 2
            state_, reward, done = env.step(action)

            state = state_
            total_step += 1
            if done:
                break
        if total_step >= 40000:
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
    # print(best_money)
    # print("Max: ", max_total)
    print("Total Costs:", env.total_cost)
    print("Total Execution Time: ", env.exe_delay)
    print("Total Energy cost: ", env.tot_energy_cost)
    print("Total Money for offloading: ", env.tot_off_cost)
    print("Offloading decisions: ", env.off_decisions)
    print("offload from edge: ", env.off_from_edge)

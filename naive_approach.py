import csv
import itertools
import random

import matplotlib.style
import matplotlib.pyplot as plt

from cloud import Cloud
from edge import Edge
from environment import Environment
from generate_states import read_state_from_file
from mobile import Mobile

matplotlib.style.use('ggplot')
import numpy as np


def get_naive_action(state, battery, cap):
    time_id, data, cpu_cycle, uplink_r, mobile_cap, edge_cap, energy = state
    mobile = Mobile(mobile_cap)
    mc, mt, me = mobile.calculate_cost_naive(cpu_cycle)
    edge = Edge(uplink_r, edge_cap)
    ec, et, ee, ep = edge.cal_total_cost_naive(data, cpu_cycle)
    # cloud = Cloud(uplink_r)
    # cc, ct, ce, cp = cloud.cal_total_cost_naive(data, cpu_cycle)

    # if energy <= battery:  # battery is greater than certain percentage, consider edge, cloud, mobile
    #     if edge_cap <= cap:
    #         action = 1 if ec < cc else 2
    #     else:
    #         action = 2
    # else:
    #     if edge_cap <= cap:
    #         action = [mc, ec, cc].index(min([mc, ec, cc]))
    #         # action += 1
    #     else:
    #         action = [mc, cc].index(min([mc, cc]))
    #         if action == 1:
    #             action += 1
    if energy <= battery:
        action = 1
    else:
        if edge_cap <= cap:
            action = [mc, ec].index(min([mc, ec]))
        else:
            action = 0

    return action


def update(episodes, battery, cap):
    global max_total
    total_step = 1
    for episode in range(episodes):
        state = env.reset()
        for t in itertools.count():
            print("Episode [%d] Iteration: %d" % (episode, t))

            time_id, data, cpu_cycle, uplink_r, mobile_cap, edge_cap, energy = state
            # print(energy)
            #
            mobile = Mobile(mobile_cap)
            mc, mt, me = mobile.calculate_total_cost(cpu_cycle)
            # print(me)
            edge = Edge(uplink_r, edge_cap)
            e_total, e_ttime, e_ptime, e_energy, e_money, _ = edge.cal_total_cost(data, cpu_cycle)
            # cloud = Cloud(uplink_r)
            # cc, _, ct, ce, cp = cloud.cal_total_cost(data, cpu_cycle)
            total_costs = [mc, e_total]
            #
            bests[total_costs.index(min(total_costs))] += 1

            #
            # max_total = max(total_costs) if max(total_costs) > max_total else max_total
            #

            total_time = [mt, e_ttime + e_ptime]
            best_time[total_time.index(min(total_time))] += 1
            total_energy = [me, e_energy]
            best_energy[total_energy.index(min(total_energy))] += 1

            if random.uniform(0, 1) > 0.02:
                action = total_costs.index(min(total_costs))
            else:
                action = random.choice([0, 1])

            # if random.uniform(0, 1) < 2.5 * (edge_cap - 0.5):
            #     action = 1
            # else:

            action = total_costs.index(min(total_costs))
            action = get_naive_action(state, battery, cap)

            #action = 0

            # data_decision.append([data, action])
            # edge_utilization_decision.append([edge_cap, action])
            # uplink_rate_decision.append([uplink_r, action])

            state_, reward, done = env.step(action)

            state = state_
            total_step += 1
            if done:
                break
        if total_step >= 100000:
            break

    print("complete")


def to_csv(filename, data):
    with open(filename, 'w', newline='') as myfile:
        csv_writer = csv.writer(myfile)
        csv_writer.writerows(data)


if __name__ == '__main__':
    num_of_episodes = 40000
    # batteries = [30, 36, 42, 48, 54, 60, 66, 72]
    # edge_cap = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # execution_list = []
    # energy_list = []
    # money_list = []
    # offload_m = []
    # offload_e = []
    # offload_c = []
    # for battery in batteries:
    #     for cap in edge_cap:
    env = Environment()
    bests = {0: 0, 1: 0, 2: 0}
    best_time = {0: 0, 1: 0, 2: 0}
    best_energy = {0: 0, 1: 0, 2: 0}
    best_money = {0: 0, 1: 0, 2: 0}
    max_total = 0
    # data_decision = []
    # edge_utilization_decision = []
    # uplink_rate_decision = []
    update(num_of_episodes, 30, 0.75)

    #         execution_list.append(env.exe_delay)
    #         energy_list.append(env.tot_energy_cost)
    #         money_list.append(env.tot_off_cost)
    #         offload_m.append(env.off_decisions[0])
    #         offload_e.append(env.off_decisions[1])
    #         offload_c.append(env.off_decisions[2])
    #
    # execution_list[:] = [x / 100000.0 for x in execution_list]
    # energy_list[:] = [x / 100000.0 for x in energy_list]
    # money_list[:] = [x / 1000.0 for x in money_list]
    # print("Execution: ", execution_list)
    # print("Energy: ", energy_list)
    # print("money: ", money_list)
    # print("mobile: ", offload_m)
    # print("edge: ", offload_e)
    # print("cloud: ", offload_c)

    print(bests)
    print("best time: ", best_time)
    print("best energy: ", best_energy)
    print(best_money)
    # print("Max: ", max_total)
    #print("Total Costs:", env.total_cost)
    # print("Total Time(S): ", (env.exe_delay + env.trans_delay) / 100009.0)
    #print("Total Execution Time(S): ", env.exe_delay / 100009.0)
    #print("Total Transmission Time(S): ", env.trans_delay / 100009.0)
    # print("Total Energy cost(J): ", (env.proc_energy + env.trans_energy) / 100009.0)
    #print("Total Proc Energy cost(J): ", env.proc_energy/ 100009.0)
    #print("Total Trans Energy cost(J): ", env.trans_energy / 100009.0)
    # print("Total Money for offloading(Cent): ", env.tot_off_cost / 1009.0)

    to_csv("data/grd_50_delay_data.csv", env.exe_delay)
    #to_csv("data/deep_trans_delay_data.csv", env.trans_delay)
    to_csv("data/grd_50_energy_data.csv", env.proc_energy)
    #to_csv("data/deep_trans_energy_data.csv", env.trans_energy)
    print("Offloading decisions: ", env.off_decisions)
    print("Offloading decisions %: ", env.off_decisions[0] * 100.0 / 100009.0, env.off_decisions[1] * 100.0 / 100009.0)
    print("offload from edge(%): ", env.off_from_edge * 100.0 / env.off_decisions[1])
    print("offload from edge: ", env.off_from_edge)

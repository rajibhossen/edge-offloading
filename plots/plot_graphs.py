import csv
import math

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats as st
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import numpy as np
import matplotlib

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# plt.rcParams.update({'font.size': 38, 'font.weight': 'bold'})
plt.rcParams.update({'font.size': 38, 'font.family': 'Times New Roman', 'font.weight': 'bold'})


def draw_picture(filename):
    # df = pd.read_csv('../data/loss-lr-0.01-b1024-rm-40k.csv')
    df = pd.read_csv(filename, header=None)
    # print(df.groupby(['Value']).size())
    # fig = px.line(df, x = 'Step', y = 'Value', title='Loss function')
    # print(df.head())
    x = df[0]
    y = df[1]
    # ysm = gaussian_filter1d(y, sigma=1.3)
    plt.figure(figsize=(10, 6))
    plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)
    plt.plot(x, y, zorder=2)
    plt.xlim(0, 170)
    plt.xlabel("Time Unit", fontweight='bold')
    plt.ylabel("Utilization", fontweight='bold')
    yticks = np.arange(0, 1.1, 0.2)
    plt.yticks(yticks)

    plt.savefig("edge_utilization.pdf", bbox_inches='tight')
    plt.show()


# draw_picture('../data/edge_trace_plot.csv')


def draw_edge_pmf():
    df = pd.read_csv("../data/edge_trace_plot.csv", header=None)
    num_bins = 30
    counts, bins = np.histogram(df[1], bins=num_bins)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2
    probs = counts / float(counts.sum())
    print(probs.sum())  # 1.0
    plt.figure(figsize=(18, 8))
    plt.xlim(0.0, 0.95)
    # plt.ylim(0.0, 0.09)
    plt.bar(bins, probs, 1.0 / num_bins, edgecolor='black')
    plt.xlabel("Utilization", fontweight='bold')
    plt.ylabel("pmf", fontweight='bold')
    plt.savefig("edge_pmf.pdf", bbox_inches='tight')
    plt.show()


# draw_edge_pmf()


def grouped_plots():
    # dqn offloading decision - {0: 41461, 1: 58548}/4228
    # naive {0: 33381, 1: 66628} / 4863
    mobile = [41461 / 1000.0, 33381 / 1000.0]
    edge = [58548 / 1000.0, 66628 / 1000.0]

    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))

    ind = np.arange(2)  # the x locations for the groups
    width = 0.2  # the width of the bars
    ax.bar(ind, mobile, width, label='M')
    ax.bar(ind + width, edge, width, label='E')
    # ax.bar(ind + width + width, cloud, width, label='C')

    # ax.set_title('Offloading decisions by algorithms')
    ax.set_xticks(ind + width / 2.0)
    ax.set_xticklabels(('DQN', 'Greedy'))
    plt.ylabel("Offloading %", fontweight='bold')
    ax.legend(loc='best')
    # ax.autoscale_view()
    plt.savefig("offload_percentage_comparison_grouped.pdf")
    plt.show()


# grouped_plots()


def draw_box_plot():
    plt.figure(figsize=(10, 5))
    plt.grid(color='#999999', linestyle='--', alpha=0.4, zorder=1)
    df1 = pd.read_csv("../data/alm_proc_delay_data.csv")
    df2 = pd.read_csv("../data/alm_trans_delay_data.csv")
    alm_delay = df1.add(df2, fill_value=0)
    df3 = pd.read_csv("../data/ale_proc_delay_data.csv")
    df4 = pd.read_csv("../data/ale_trans_delay_data.csv")
    ale_delay = df3.add(df4, fill_value=0)
    grd_delay = pd.read_csv("../data/grd_20_delay_data.csv")
    opt_delay = pd.read_csv("../data/opt_20_delay_data.csv")
    deep_delay = pd.read_csv("../data/deep_20_delay_data.csv")
    # df5 = pd.read_csv("../data/grd_proc_energy_data.csv")
    # df6 = pd.read_csv("../data/grd_trans_energy_data.csv")
    # grd_delay = df5.add(df6, fill_value=0)
    # df7 = pd.read_csv("../data/opt_proc_energy_data.csv")
    # df8 = pd.read_csv("../data/opt_trans_energy_data.csv")
    # opt_delay = df7.add(df8, fill_value=0)
    # df9 = pd.read_csv("../data/deep_proc_energy_data.csv")
    # df10 = pd.read_csv("../data/deep_trans_energy_data.csv")
    # deep_delay = df9.add(df10, fill_value=0)

    data = pd.concat([alm_delay, ale_delay, grd_delay, opt_delay, deep_delay], axis=1)

    data.boxplot(showfliers=False, whis=[10, 90], showmeans=True, whiskerprops=dict(linewidth=3.0),
                 boxprops=dict(linewidth=3.0),
                 medianprops=dict(linewidth=3.0),
                 meanprops=dict(marker='d', markersize=18.0, markeredgewidth=3, markeredgecolor='black', markerfacecolor='yellow'))
    plt.ylabel("Avg. Task \n Completion Time (s)", fontweight='bold')
    #plt.ylabel("Avg. Energy (J)", fontweight='bold')
    # yticks = np.arange(1.2, 1.5, 0.1)
    yticks = np.arange(1, 6, 1)
    plt.yticks(yticks)
    #plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=1.0)
    plt.savefig("execution_delay_boxplot.pdf", bbox_inches='tight')
    plt.show()


draw_box_plot()


def plot_error_bars():
    # for i in range(1, 6):
    #     df = pd.read_csv("../data/grd_" + str(i * 10) + "_energy_data.csv", header=None)
    #     confidence_level = 0.95
    #     degrees_freedom = df.size - 1
    #     sample_mean = df.mean()
    #     sample_standard_error = df.std()
    #     confidence_interval = st.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    #     print("filename: ", i*10)
    #     print(confidence_interval)
    algo = ['10', '20', "30", "40", "50"]

    best_exec_cost = [1.19597, 1.229501, 1.27119, 1.320903, 1.366759]
    best_energy_cost = [2.066611, 2.062006, 2.06848, 2.083136, 2.102163]
    best_exec_ci = [(0.22015715, 2.17178364), (0.15928237, 2.29972013), (0.09473241, 2.44764829),
                    (0.03938555, 2.60242121), (0.00671844, 2.72679868)]
    best_energy_ci = [(1.04418083, 3.08904148), (1.05475838, 3.06925407), (1.0429976, 3.09396178),
                      (1.01545602, 3.1508157), (0.98288196, 3.22144336)]
    best_exec_r = [best_exec_cost[i] - best_exec_ci[i][1] for i in range(len(best_exec_ci))]
    best_energy_r = [best_energy_cost[i] - best_energy_ci[i][1] for i in range(len(best_energy_ci))]

    dqn_exec_cost = [1.205513, 1.238632, 1.280195, 1.329607, 1.377893]
    dqn_energy_cost = [2.073121, 2.070685, 2.075284, 2.089615, 2.106603]
    dqn_exec_ci = [(0.17961299, 2.2314137), (0.1229479, 2.35431568), (0.07045898, 2.48993077), (0.02098094, 2.63823236),
                   (-0.01233297, 2.76811814)]
    dqn_energy_ci = [(1.03338486, 3.11285664), (1.03934044, 3.10202935), (1.03118254, 3.11938543),
                     (1.00509308, 3.17413665), (0.97849451, 3.23471134)]
    dqn_exec_r = [dqn_exec_cost[i] - dqn_exec_ci[i][1] for i in range(len(dqn_exec_ci))]
    dqn_energy_r = [dqn_energy_cost[i] - dqn_energy_ci[i][1] for i in range(len(dqn_energy_ci))]

    grd_exec_cost = [1.234449, 1.271246, 1.319452, 1.383064, 1.448965]
    grd_energy_cost = [2.089169, 2.089169, 2.089169, 2.089169, 2.089169]
    grd_exec_ci = [(0.09246705, 2.37643052), (0.03782738, 2.50466389), (-0.01482353, 2.65372733),
                   (-0.06203821, 2.8281669), (-0.0855402, 2.98346995)]
    grd_energy_ci = [(1.02885853, 3.14947938), (1.02885853, 3.14947938), (1.02885853, 3.14947938),
                     (1.02885853, 3.14947938), (1.02885853, 3.14947938)]
    grd_exec_r = [grd_exec_cost[i] - grd_exec_ci[i][1] for i in range(len(grd_exec_ci))]
    grd_energy_r = [grd_energy_cost[i] - grd_energy_ci[i][1] for i in range(len(grd_energy_ci))]

    #
    ind = np.arange(7, 55, 10)
    print(ind)
    width = 3  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)

    ax.bar(ind, grd_energy_cost, width, edgecolor='white', label='GRD', zorder=2, yerr=grd_energy_r)
    ax.bar(ind + width, best_energy_cost, width, edgecolor='white', label='OPT', zorder=2, yerr=best_energy_r)
    ax.bar(ind + width + width, dqn_energy_cost, width, edgecolor='white', label='DeepTO', zorder=2, yerr=dqn_energy_r)
    # ax.set_ylabel('Avg. Task \nCompletion Time (s)', fontweight='bold')
    ax.set_ylabel('Avg. Energy (J)', fontweight='bold')
    ax.set_ylim(0, 4.1)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels(algo)
    ax.legend(loc='upper right', prop={'size': 26}, handlelength=1.2, ncol=3)
    plt.savefig('energy_used_ci.pdf', bbox_inches='tight')
    plt.show()


# plot_error_bars()


def plot_lines():
    base_ofe = [15.71, 20.79, 29.78, 40.26, 50.09]

    # best_exec_cost = [1.195983744500096, 1.276996853290992, 1.3466684652181335, 1.4039901608086027, 1.4554154350275383]
    # best_energy_cost = [2.062681030657753, 2.115646880057367, 2.1610913036935533, 2.191114225452612, 2.208561435164212]
    # best_money = [0.8132566557945994, 0.7810958764129147, 0.7203357918249896, 0.6526418582748283, 0.5985414020508861]
    #
    # dqn_exec_cost = [1.2054434152193767, 1.287108509150544, 1.3573838439415042, 1.4136038753630014, 1.4643613811219902]
    # dqn_energy_cost = [2.068159031899507, 2.122975537568714, 2.1648811535428267, 2.1937266479844073, 2.213954629886179]
    # dqn_money = [0.8072391854751549, 0.7761492051851208, 0.7171100448444026, 0.6534783873710904, 0.5991466479606599]
    #
    # greedy_exec = [1.2517385614388996, 1.3651562205799526, 1.4741627964313864, 1.567818366098802, 1.6654726983560901]
    # greedy_energy = [2.0786094605839036, 2.152960481933771, 2.23002300219896, 2.3000065754727292, 2.368329566540824]
    # greedy_money = [0.8779512955058899, 0.8438012916430067, 0.7670832266876746, 0.678519703933839, 0.5994083027265401]

    best_exec_cost = [1.2101487780542226, 1.2287008271347708, 1.270186773078713, 1.3227986255855244,
                      1.368276681640847]
    best_energy_cost = [2.062643828289949, 2.0624354022102174, 2.0687857846057387, 2.083617189122425,
                        2.100994816841319]
    # best_money = [0.7721407790209025, 0.7204196267342626, 0.640540414557095, 0.5495480971755499,
    #               0.47360694792420543]

    dqn_exec_cost = [1.2195413597615195, 1.2387325240023765, 1.2801289206489315, 1.3293352156791616,
                     1.3785932100843086]
    dqn_energy_cost = [2.070329254213621, 2.0685622435800397, 2.076088903185661, 2.0896139708749675,
                       2.1066175168131016]
    # dqn_money = [0.7667433160995489, 0.7178551604897137, 0.634799586536289, 0.5501716367975693,
    #              0.4712041354019441]

    greedy_exec = [1.2499513552584505, 1.2718562403349378, 1.319839992193188, 1.3815806239942854,
                   1.4508628132655321]
    greedy_energy = [2.0893569806881582, 2.0893569806881582, 2.0893569806881582, 2.0893569806881582,
                     2.0893569806881582]
    # greedy_money = [0.8002144886324871, 0.7570104920628667, 0.6851462851185408, 0.6057551266326865,
    #                 0.53067579336764]

    plt.figure(figsize=(10, 5))
    plt.grid(color='#999999', linestyle='--', alpha=0.4, zorder=1)

    plt.plot(base_ofe, greedy_energy, linestyle='-', linewidth=3, marker="d", markersize=18, markerfacecolor='yellow', markeredgewidth=3,
             zorder=2)
    plt.plot(base_ofe, best_energy_cost, linestyle='--', linewidth=3, marker="s", markersize=18, markeredgewidth=3,
             markerfacecolor='yellow', zorder=2)
    plt.plot(base_ofe, dqn_energy_cost, linestyle='--', linewidth=3, marker="o", markersize=18, markeredgewidth=3,
             markerfacecolor='yellow',
             zorder=2)

    plt.legend(['GRD', 'OPT', 'DeepTO'], loc='upper left', ncol=3, prop={'size': 26}, handlelength=1.2)
    plt.xlabel("% Offloaded from Edge to Cloud", fontweight='bold')
    #plt.ylabel("Avg. Task \nCompletion Time (s)", fontweight='bold')
    plt.ylabel("Avg. Energy (J)", fontweight='bold')
    # xticks = np.arange(15, 55, 12)
    # plt.xticks(xticks)
    # plt.xlim(15, 52)

    yticks = np.arange(1.2, 1.5, 0.1) # for exec
    yticks = np.arange(2.06, 2.15, 0.03) # for energy
    plt.yticks(yticks)
    plt.savefig('energy_used_var_camera.pdf', bbox_inches='tight')
    plt.show()


#plot_lines()


def alternate_plot_bars():
    plt.figure(figsize=(11, 5))
    algo = ['DQN', 'Greedy', 'A-E', 'A-C', 'A-M']
    # {0: 28400, 1: 52008, 2: 19607} / 426, naive-{0: 18742, 1: 69006, 2: 12257}/1158
    execution_delay = [117037.33088242248, 121127.02432451918, 131980.07118882937, 164573.16829485915,
                       178230.71372128194]
    total_energies = [204631.4237137091, 208177.87750245916, 199333.42912758567, 227456.044875685, 319707.7749986188]
    money = [680.9484264278958, 847.8617202768415, 1205.5920450872495, 369.29792633961705, 0]

    execution_delay[:] = [x / 100000.0 for x in execution_delay]
    total_energies[:] = [x / 100000.0 for x in total_energies]
    money[:] = [x / 100000.0 for x in money]
    plt.bar(algo, money)
    plt.ylabel("Cost ($)", fontweight='bold')
    plt.savefig("alt_offload_price.pdf", bbox_inches='tight')
    plt.show()


# alternate_plot_bars()


def plot_bars():
    # plt.figure(figsize=(10, 6))
    # plt.grid(color='#999999', linestyle='-.', alpha=0.3, zorder=-1)
    # plt.grid(color='#999999', linestyle='-.', alpha=0.3)
    # ax = fig.add_axes([0, 0, 1, 1])
    algo = ['ALM', 'ALE', 'GRD', 'OPT', 'DeepTO']
    # algo = ['ALE', 'GRD', "DeepTO"]

    execution_delay = [1.906902597595083, 0.6192736978930504, 0.8077430433341862, 0.7884015470576238,
                       0.7967345944612377]
    transmission_delay = [0.0000000000, 0.8555785603223353, 0.4652141358417224, 0.44031688774452593,
                          0.44080438475925293]
    processing_energy = [2.8139227683981107, 0.0000000000, 0.8608437764433042, 0.9098808266304683, 0.9174520367418839]
    transmission_energy = [0.00000000000, 1.9935152377489536, 1.2285132042448539, 1.153607533981035, 1.1508997809788128]
    money = [0.0000000000000000, 1.0647551891926474, 0.7565275715519929, 0.7214848115061836, 0.7189953286753467]

    # offloading_from_edge = [20815 * 100 / 100009.0, 7636 * 100 / 59827.0, 7871.0 * 100 / 53176.0]
    offloading_from_edge = [39864.0 * 100 / 100009.0, 19548 * 100 / 61852.0, 17546 * 100 / 55189.0]
    #
    # dqn offloading decision - {0: 46825, 1: 53176} / 7871
    # naive {0: 33887, 1: 66120} / 9627

    # plt.bar(algo, offloading_from_edge)
    # plt.ylabel("Offloading %", fontweight='bold')
    # # plt.savefig("offload_from_edge_comparison.pdf", bbox_inches='tight')
    # plt.show()

    width = 0.5  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)

    ax.bar(algo, processing_energy, width, edgecolor='white', label='Processing', zorder=2)
    ax.bar(algo, transmission_energy, width, edgecolor='white', bottom=np.array(processing_energy),
           label='Transmission', zorder=2)
    ax.set_ylabel('Avg. Energy (J)', fontweight='bold')
    ax.set_ylim(0, 3.5)

    ax.legend(loc='upper right', prop={'size': 32}, handlelength=1.2)
    plt.savefig('energy_used_stack_temp.pdf', bbox_inches='tight')
    plt.show()


# plot_bars()


def draw_weight_impact():
    w1_exe = [1.3019527787068281, 1.2817815168472433, 1.2511118193232513, 1.2048594798413295, 1.1808345766283912,
              1.1738474774830343, 1.1715753627514645]
    w1_energy = [1.9705193309230309, 1.9816042435708787, 2.003027196551032, 2.0691582481926996, 2.1405897286155384,
                 2.174759097004665, 2.1935538010031888]
    w1_money = [0.8701535160995516, 0.8675511067506682, 0.856220077970414, 0.8105383881796483, 0.7517187365399876,
                0.7281692186932294, 0.7167307857394873]
    weight1 = ["0.38", "0.5", "0.75", "1.5", "3", "4.5", "6"]
    w1_exe[:] = [100 * (x / 1.2048594798413295 - 1) for x in w1_exe]
    w1_energy[:] = [100 * (x / 2.0691582481926996 - 1) for x in w1_energy]
    w1_money[:] = [100 * (x / 0.8105383881796483 - 1) for x in w1_money]

    weight2 = ["0.25", "0.35", "0.5", "1", "2", "3", "4"]
    w2_exe = [1.1801247460822701, 1.1784998223364422, 1.1768896109852969, 1.2047975579298056, 1.2671396704331812,
              1.2973380838140025, 1.3158182369732019]
    w2_energy = [2.333145958848733, 2.293733176683389, 2.231745270211371, 2.069093384327998, 1.949413446788055,
                 1.9190381847676763, 1.9066947871822264]
    w2_money = [0.5526478900073223, 0.5878365901674335, 0.6433227239563504, 0.8104541774326416, 0.9566853857519023,
                0.9971460956047159, 1.0132147982401136]
    w2_exe[:] = [100 * (x / 1.2047975579298056 - 1) for x in w2_exe]
    w2_energy[:] = [100 * (x / 2.069093384327998 - 1) for x in w2_energy]
    w2_money[:] = [100 * (x / 0.8104541774326416 - 1) for x in w2_money]

    weight3 = ["12", "20", "25", "50", "100", "150", "200"]
    w3_exe = [1.2298991458535047, 1.2234842007065818, 1.2204656286009985, 1.2033036773939447, 1.2014346123441109,
              1.2357007478098845, 1.2886468487045732]
    w3_energy = [1.9903000837542297, 2.00469498608417, 2.0127185677413917, 2.0688023467670447, 2.213202437170408,
                 2.347855605163478, 2.455917289179239]
    w3_money = [0.9355692647565166, 0.9112971788737931, 0.8954412588397161, 0.811152176074586, 0.6258975444986463,
                0.4802874871770619, 0.37092179576845885]
    w3_exe[:] = [100 * (x / 1.2033036773939447 - 1) for x in w3_exe]
    w3_energy[:] = [100 * (x / 2.0688023467670447 - 1) for x in w3_energy]
    w3_money[:] = [100 * (x / 0.811152176074586 - 1) for x in w3_money]

    plt.figure(figsize=(10, 6))
    # plt.bar(weight3, w3_money)
    legends = ["Latency", "Energy", "Cost"]
    plt.plot(weight2, w2_exe, linestyle="-", linewidth=3, markersize=12, marker=">")
    plt.plot(weight2, w2_energy, linestyle="--", linewidth=3, markersize=12, marker="v")
    plt.plot(weight2, w2_money, linestyle="dashdot", linewidth=3, markersize=12, marker="^")
    plt.ylabel("% Change", fontweight='bold')
    plt.xlabel("$W_2$", fontweight='bold')
    plt.legend(legends, loc='lower left', fontsize=28)
    plt.savefig("weight2_impacts.pdf", bbox_inches='tight')
    plt.show()


# draw_weight_impact()


def draw_weight_decisions():
    w1_mobile = [28421, 28822, 29245, 29922, 30742, 31714, 32666]
    w1_edge = [30655, 35453, 39563, 42790, 44719, 46074, 46908]
    w1_cloud = [40924, 35740, 31203, 27299, 24550, 22223, 20437]
    w1_mobile[:] = [x / 1000.0 for x in w1_mobile]
    w1_edge[:] = [x / 1000.0 for x in w1_edge]
    w1_cloud[:] = [x / 1000.0 for x in w1_cloud]
    weight1 = ["0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75"]

    w2_m = [54015, 40805, 33551, 29965, 27687, 26328, 25278]
    w2_e = [23021, 31182, 37507, 42706, 47137, 51033, 54484]
    w2_c = [22978, 28014, 28953, 27340, 25187, 22639, 20238]
    w2_m[:] = [x / 1000.0 for x in w2_m]
    w2_e[:] = [x / 1000.0 for x in w2_e]
    w2_c[:] = [x / 1000.0 for x in w2_c]

    w3_m = [21745, 23249, 25811, 29918, 33646, 36991, 40020]
    w3_e = [68680, 64853, 57289, 42561, 30232, 21532, 14568]
    w3_c = [9575, 11889, 16900, 27532, 36133, 41488, 45423]
    w3_m[:] = [x / 1000.0 for x in w3_m]
    w3_e[:] = [x / 1000.0 for x in w3_e]
    w3_c[:] = [x / 1000.0 for x in w3_c]
    weight3 = ["1", "10", "25", "50", "75", "100", "125"]

    width = 0.5  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(weight1, w2_m, width, edgecolor='white', label='M')
    ax.bar(weight1, w2_e, width, edgecolor='white', bottom=np.array(w2_m), label='E')
    ax.bar(weight1, w2_c, width, edgecolor='white', bottom=np.array(w2_m) + np.array(w2_e), label='C')
    ax.set_ylim(0, 130)
    ax.set_ylabel('Percent', fontweight='bold')
    ax.set_xlabel("$W_2$")
    ax.legend(loc='upper center', ncol=3, fontsize=28)
    plt.savefig('weight2_decisions.pdf', bbox_inches='tight')
    plt.show()


# draw_weight_decisions()


def draw_losses():
    # sns.set_style("whitegrid")
    # fig,ax = plt.subplots()
    # legends = []
    # files = glob.glob("../data/reward-lr-0.001-*.csv")
    with open("../data/job_arrival_times.csv", newline='') as file:
        reader = csv.reader(file)
        job_arrivals = list(reader)

    # print(math.floor(float(job_arrivals[math.floor(2.5)*18][0])))
    files = ['../data/loss-1e-3-128.csv', '../data/loss-1e-5-128.csv', '../data/loss-1e-6-128.csv']
    # files = ['../data/loss-pre-trained.csv']
    x = []
    y = []
    plt.figure(figsize=(10, 6))
    plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)
    plt.xlim(0, 5000)
    # plt.ylim(0, 300)
    for i, file in enumerate(files):
        # draw_picture(file)
        df = pd.read_csv(file)
        df1 = df[:1000]
        x.append(df1['Step'])
        y1 = df1['Value']
        ysm = gaussian_filter1d(y1, sigma=3)
        y.append(ysm)

    plt.plot(x[0], y[0], linestyle='-.', label='1e-3', linewidth=3, zorder=2)
    plt.plot(x[1], y[1], linestyle='-', label='1e-5', linewidth=3, zorder=2)
    plt.plot(x[2], y[2], linestyle='--', label='1e-6', linewidth=3, zorder=2)
    # plt.plot(x[3], y[3], linestyle='-', label='1e-6', linewidth=4)
    # plt.plot(x[4], y[4], linestyle='-.', label='Cloud')

    plt.legend(loc="upper left", ncol=3, prop={'size': 26}, handlelength=1.3)
    # plt.ylim(0, 400)
    plt.xlabel("Episodes", fontweight='bold')
    plt.ylabel("Value", fontweight='bold')
    xticks = np.arange(0, 6000, 1000)
    plt.xticks(xticks)
    yticks = np.arange(0, 450, 100)
    plt.yticks(yticks)
    # plt.ylim(0,400)
    # plt.title("Loss Function")
    # plt.savefig("loss_comparison_3_6.pdf", bbox_inches='tight')
    plt.show()

# draw_picture()
# draw_comparisons()
# state_generation()
# draw_losses()
# plot_bars()
# grouped_plots()

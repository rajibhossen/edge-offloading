import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns


def draw_picture(filename):
    # df = pd.read_csv('../data/loss-lr-0.01-b1024-rm-40k.csv')
    df = pd.read_csv(filename)
    print(df.groupby(['Value']).size())
    # fig = px.line(df, x = 'Step', y = 'Value', title='Loss function')
    x = df['Step']
    y = df['Value']
    ysm = gaussian_filter1d(y, sigma=1.3)

    plt.plot(x, ysm)
    plt.xlabel("Steps")
    plt.ylabel("Action")
    plt.title("Loss function")
    # plt.ylabel("Average Reward")
    # plt.title("Average Reward Over Episodes")
    plt.show()


def plot_bars():
    fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    algo = ['DQN', 'Greedy', 'All Edge', 'All Mobile', 'All Cloud']
    # algo = ['Mobile', 'edge', 'cloud']
    # total execution data

    total_costs = [171046.9643922055, 166156.76108757418, 171522.4394385646, 359041.878701425, 189282.42264744863]
    execution_delay = [246932.48422193376, 250630.36288031228, 254668.19884885458, 350457.72138229763,
                       362504.1459842049]
    total_energies = [68930.47869846618, 75408.20467287282, 69039.00180737283, 271427.4483558506, 91565.64354255877]
    money = [2878.45051748448, 2977.367577309831, 3881.638791897804, 0, 709.0742608838378]

    # total_costs = [326249.3218534179, 621885.1697381499, 327568.9100616688, 454383.4887876519, 327164.0669541478]
    # execution_delay = [246932.48422193376, 350457.72138229763, 254652.81151009543, 362108.77098420705, 250630.36288031228]
    # total_energies = [75961.4927169943, 271427.4483558506, 69039.00180737283, 91565.64354255877, 73364.95670569877]
    # money = [3355.344914489943, 0, 3877.0967442000406, 709.0742608838378, 3168.747368135734]
    execution_delay[:] = [x / 100000.0 for x in execution_delay]
    total_energies[:] = [x / 100000.0 for x in total_energies]
    money[:] = [x / 100000.0 for x in money]
    total_costs[:] = [x / 100000.0 for x in total_costs]
    weights = []
    # print(execution_delay)
    # print(total_energies)
    # print(money)
    # print(total_costs)
    plt.bar(algo, total_energies)
    plt.xlabel("Algorithms")
    plt.ylabel("Average Energy Cost (J)")
    plt.title("Average Energy Comparison of Algorithms")
    plt.savefig("energy-cost-comparison.png")
    # plt.ylabel("Total Offloading Cost (USD)")
    # plt.title("Total Offloading Cost Comparison of Algorithms")
    # plt.savefig("off-price-compariso1.png")
    plt.show()


plot_bars()


def draw_comparisons():
    sns.set_style("darkgrid")
    fig = plt.figure()
    # legends = []
    # files = glob.glob("../data/reward-lr-0.001-*.csv")
    files = ['../data/reward-rms-1e-3.csv', '../data/all-mobile-avg-reward.csv', '../data/all-edge-avg-reward.csv',
             '../data/all-cloud-avg-reward.csv']  # , '../data/naive_approach.csv']
    legends = ['LSTM', 'Mobile', 'Edge', 'Cloud']

    x = []
    y = []
    for i, file in enumerate(files):
        # draw_picture(file)
        df = pd.read_csv(file)
        x.append(df['Step'])
        y1 = df['Value']
        ysm = gaussian_filter1d(y1, sigma=1.3)
        y.append(ysm)
        # lr = re.search('lr-(.+?)-b', file)
        # rm = re.search('rm-(.+?)k', file)
        # lr = lr.group(1)
        # rm = rm.group(1)
        # legends.append(str(lr) + "/" + str(rm))
        # # plt.legend("lr-" + str(m.group(1)))
        # sns.lineplot(x,ysm)

    plt.plot(x[0], y[0], linestyle='-.', label='DQN')
    plt.plot(x[1], y[1], linestyle='-', label='Mobile')
    plt.plot(x[2], y[2], linestyle='--', label='Edge')
    plt.plot(x[3], y[3], linestyle='-', label='Cloud')
    # plt.plot(x[4], y[4], linestyle='-.', label='Naive')

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.title("Average Reward")
    # plt.savefig("average-rewards-comparison.png")
    plt.show()


def draw_losses():
    sns.set_style("darkgrid")
    fig = plt.figure()
    # legends = []
    # files = glob.glob("../data/reward-lr-0.001-*.csv")
    files = ['../data/loss-1e-3.csv', '../data/loss-1e-4.csv', '../data/loss-1e-6.csv']

    x = []
    y = []
    for i, file in enumerate(files):
        # draw_picture(file)
        df = pd.read_csv(file)
        df1 = df[:1000]
        print(df1.shape)
        x.append(df1['Step'])
        y1 = df1['Value']
        ysm = gaussian_filter1d(y1, sigma=1.3)
        y.append(ysm)
        # lr = re.search('lr-(.+?)-b', file)
        # rm = re.search('rm-(.+?)k', file)
        # lr = lr.group(1)
        # rm = rm.group(1)
        # legends.append(str(lr) + "/" + str(rm))
        # # plt.legend("lr-" + str(m.group(1)))
        # sns.lineplot(x,ysm)

    plt.plot(x[0], y[0], linestyle='-.', label='1e-3')
    plt.plot(x[1], y[1], linestyle='-', label='1e-4')
    plt.plot(x[2], y[2], linestyle='--', label='1e-6')
    # plt.plot(x[3], y[3], linestyle='-', label='0.00001-100')
    # plt.plot(x[4], y[4], linestyle='-.', label='Cloud')

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.title("Loss Function")
    plt.savefig("loss-comparison.png")
    plt.show()

# draw_picture()
# draw_comparisons()
# state_generation()
# draw_losses()

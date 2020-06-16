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
    algo = ['DQN', 'All Mobile', 'All Edge', 'All Cloud', 'Naive']
    # total execution data
    execution_delay = [2628303.5473039835, 13518874.000179151, 3142864.171568715, 2323662.949356631, 1830121.6529592853]
    total_energies = [621120.9701678093, 1749465.1073236275, 344369.881719523, 456869.8817195224, 384392.8817195243]
    money = [200000, 0, 540183.1558762271, 191984.009469998, 224254.14633212643]
    missed_deadlines = [9751, 19221, 29641, 14280, 5895]
    plt.bar(algo, money)
    plt.xlabel("Algorithms")
    plt.ylabel("Total Delays")
    plt.title("Execution Time Comparison of Algorithms")
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
    files = ['../data/reward.csv', '../data/reward-r.csv']

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

    plt.plot(x[0], y[0], linestyle='-.', label='1e-4')
    plt.plot(x[1], y[1], linestyle='-', label='1e-4 v2')
    # plt.plot(x[2], y[2], linestyle='--', label='1e-5')
    # plt.plot(x[3], y[3], linestyle='-', label='0.00001-100')
    # plt.plot(x[4], y[4], linestyle='-.', label='Cloud')

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.title("Loss Function")
    # plt.savefig("loss-comparison.png")
    plt.show()


def state_generation():
    df1 = pd.read_csv('../data/state-trace.csv')
    df2 = pd.read_csv('../data/state-trace-0.01.csv')
    df3 = pd.read_csv('../data/state-trace-0.01-n.csv')
    df4 = pd.read_csv('../data/state-trace-0.001.csv')
    df5 = pd.read_csv('../data/state-trace-0.001-n.csv')
    df6 = pd.concat([df1, df2, df3, df4, df5], sort=False)
    # df.to_csv('../data/states.csv', index=False)
    df6 = df6.drop_duplicates()
    print(df6.shape)
    df6.to_csv("../data/states.csv", header=False, index=False)

# draw_picture()
# draw_comparisons()
# state_generation()
#draw_losses()

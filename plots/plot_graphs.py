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
    algo = ['All Mobile', 'All Edge', 'All Cloud', 'Naive']
    # total execution data
    execution_delay = [13518874.000179151, 3055013.0161278867, 3205603.9454013384, 3062494.8857348566]
    total_energies = [634557.0049511091, 1749465.1073236275, 344369.881719523, 456869.8817195224, 4267063.307657191]
    money = [309192.8942792864, 0, 543018.6633310195, 191984.009469998, 224254.14633212643]
    missed_deadlines = [9751, 20837, 29641, 14280, 5895]
    execution_delay[:] = [x / 100000.0 for x in execution_delay]

    plt.bar(algo, execution_delay)
    plt.xlabel("Algorithms")
    plt.ylabel("Total Execution Delay (Sec)")
    plt.title("Total Execution Delay Comparison of Algorithms")
    # plt.savefig("execution-time-comparison.png")
    # plt.ylabel("Total Offloading Cost (USD)")
    # plt.title("Total Offloading Cost Comparison of Algorithms")
    plt.savefig("off-price-comparison1.png")
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
    files = ['../data/loss-r.csv', '../data/loss-rms-1e-5.csv']

    x = []
    y = []
    for i, file in enumerate(files):
        # draw_picture(file)
        df = pd.read_csv(file)
        df1 = df[:1500]
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
    plt.plot(x[1], y[1], linestyle='-', label='1e-5')
    # plt.plot(x[2], y[2], linestyle='--', label='1e-5')
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

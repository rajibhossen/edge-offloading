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
    algo = ['DQN', 'All Mobile', 'All Edge', 'All Cloud', 'Greedy']
    # algo = ['Mobile', 'edge', 'cloud']
    # total execution data
    total_costs = [184311.45198231575, 359041.878701425, 171522.4394385646, 189282.42264744863, 166156.76108757418]
    execution_delay = [266891.992916002, 350457.72138229763, 254668.19884885458, 362504.1459842049, 259899.52256641316]
    total_energies = [84003.29122426435, 271427.4483558506, 69039.00180737283, 91565.64354255877, 75408.20467287282]
    money = [3358.516252905033, 0, 3881.638791897804, 709.0742608838378, 2577.367577309831]
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
    plt.ylabel("Average Execution Delay (Sec)")
    plt.title("Average Execution Delay Comparison of Algorithms")
    # plt.savefig("execution-delay-comparison.png")
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
    files = ['../data/loss-1e-4.csv', '../data/loss-1e-7.csv']

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
    plt.plot(x[1], y[1], linestyle='-', label='1e-7')
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

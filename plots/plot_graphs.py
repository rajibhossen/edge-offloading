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


# draw_picture("../data/dqn-episode-actions.csv")

sns.set_style("darkgrid")
fig = plt.figure()
# legends = []
# files = glob.glob("../data/reward-lr-0.001-*.csv")
files = ['../data/reward_avg-dqn-tf.csv', '../data/naive_approach.csv',
         '../data/all-mobile-avg-reward.csv',
         '../data/all-edge-avg-reward.csv', '../data/all-cloud-avg-reward.csv']
legends = ['DQN', 'Naive', 'Mobile', 'Edge', 'Cloud']

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
plt.plot(x[1], y[1], linestyle='-', label='Naive')
plt.plot(x[2], y[2], linestyle='--', label='Mobile')
plt.plot(x[3], y[3], linestyle='-', label='Edge')
plt.plot(x[4], y[4], linestyle='-.', label='Cloud')

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.title("Average Reward")
# plt.savefig("average-rewards-comparison.png")
plt.show()

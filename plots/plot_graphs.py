import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d


def draw_picture(filename):
    # df = pd.read_csv('../data/loss-lr-0.01-b1024-rm-40k.csv')
    df = pd.read_csv(filename)
    print(df.groupby(['Value']).size())
    # fig = px.line(df, x = 'Step', y = 'Value', title='Loss function')
    # x = df['Step']
    # y = df['Value']
    # ysm = gaussian_filter1d(y, sigma=1.3)
    #
    # plt.plot(x, ysm)
    # plt.xlabel("Steps")
    # plt.ylabel("Action")
    # plt.title("Loss function")
    # # plt.ylabel("Average Reward")
    # # plt.title("Average Reward Over Episodes")
    # plt.show()


draw_picture("../data/dqn-step-actions.csv")

fig = plt.figure()
# legends = []
# files = glob.glob("../data/reward-lr-0.001-*.csv")
files = ['../data/reward_avg-lr-0.001-b1024-rm-10k.csv', '../data/all-mobile-avg-reward.csv',
         '../data/all-edge-avg-reward.csv', '../data/all-cloud-avg-reward.csv']
legends = ['DQN', 'Mobile', 'Edge', 'Cloud']
for file in files:
    # draw_picture(file)
    df = pd.read_csv(file)
    x = df['Step']
    y = df['Value']
    ysm = gaussian_filter1d(y, sigma=1.3)
    # lr = re.search('lr-(.+?)-b', file)
    # rm = re.search('rm-(.+?)k', file)
    # lr = lr.group(1)
    # rm = rm.group(1)
    # legends.append(str(lr) + "/" + str(rm))
    # # plt.legend("lr-" + str(m.group(1)))
    plt.plot(x, ysm)
plt.legend(legends)
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.title("Average function")
plt.show()

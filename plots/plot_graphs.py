import re

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import glob


def draw_picture(filename):
    # df = pd.read_csv('../data/loss-lr-0.01-b1024-rm-40k.csv')
    df = pd.read_csv(filename)

    # fig = px.line(df, x = 'Step', y = 'Value', title='Loss function')
    x = df['Step']
    y = df['Value']
    ysm = gaussian_filter1d(y, sigma=1.3)

    plt.plot(x, ysm)
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.title("Loss function")
    # plt.ylabel("Average Reward")
    # plt.title("Average Reward Over Episodes")
    plt.show()


fig = plt.figure()
legends = []
files = glob.glob("../data/accuracy-lr-0.001-*.csv")
for file in files:
    # draw_picture(file)
    df = pd.read_csv(file)
    x = df['Step']
    y = df['Value']
    ysm = gaussian_filter1d(y, sigma=2)
    lr = re.search('lr-(.+?)-b', file)
    rm = re.search('rm-(.+?)k', file)
    lr = lr.group(1)
    rm = rm.group(1)
    legends.append(str(lr) + "/" + str(rm))
    # plt.legend("lr-" + str(m.group(1)))
    plt.plot(x, ysm)
plt.legend(legends)
plt.xlabel("Episodes")
plt.ylabel("Value")
plt.title("Loss function")
plt.show()

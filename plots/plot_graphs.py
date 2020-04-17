import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# df = pd.read_csv('../data/loss-lr-0.01-b1024-rm-40k.csv')
df = pd.read_csv('../data/reward_avg-lr-0.01-b1024-rm-40k.csv')

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

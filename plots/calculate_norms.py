import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

x_axis = []
norms = []

for i in range(1, 21):
    filename = "../data/episodes-" + str(i * 1000) + ".csv"
    # if i == 1:
    #     q_table = pd.read_csv("../data/episodes-500.csv", index_col=0)
    #     q_table = q_table.to_numpy()
    #     norm = np.linalg.norm(q_table)
    #     x_axis.append(i * 500)
    #     norms.append(norm)
    q_table = pd.read_csv(filename, index_col=0)
    q_table = q_table.to_numpy()
    norm = np.linalg.norm(q_table)
    x_axis.append(i*1000)
    norms.append(norm)

for i in range(3, 11):
    filename = "../data/episodes-" + str(i * 10000) + ".csv"
    q_table = pd.read_csv(filename, index_col=0)
    q_table = q_table.to_numpy()
    norm = np.linalg.norm(q_table)
    x_axis.append(i*10000)
    norms.append(norm)

print(x_axis)
print(norms)
plt.plot(x_axis, norms)
#plt.ylim(0,30000)
plt.xlabel("iteration")
plt.ylabel("2-norm")
plt.savefig("norms-vs-episodes.png")
plt.show()


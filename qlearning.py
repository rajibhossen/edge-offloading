"""
@project: Mobile Edge Offloading
@author: Md Rajib Hossen
@time: 03/15/2020
@email: mdrajib.hossen@mavs.uta.edu
"""
import numpy as np
import pandas as pd
import os


class QLearningTable:
    def __init__(self, actions, filename="", lr=0.01, discount=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = lr
        self.discount = discount
        self.epsilon = e_greedy
        # If the specified file is not empty, then read the Q table from the specified file
        if filename != "":
            self.q_table = pd.read_csv(filename, index_col=0)
            self.q_table.columns = list(range(0, len(self.actions)))
        else:
            if os.path.exists("data/q_table.csv"):
                if os.stat("data/q_table.csv").st_size == 0:
                    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
                else:
                    self.q_table = pd.read_csv("data/q_table.csv", index_col=0)
                    self.q_table.columns = list(range(0, len(self.actions)))
            else:
                self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exists(state)
        # action selection trade of between exploration and exploitation, explore 10% of the time
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, c_state, action, reward, n_state):
        self.check_state_exists(n_state)
        q_predict = self.q_table.loc[c_state, action]

        n_state_ls = list(n_state.split(","))

        if n_state_ls[1] != " -1":
            q_target = reward + self.discount * self.q_table.loc[n_state, :].max()
        else:
            print("reward only")
            q_target = reward

        # q_target = reward + self.discount * self.q_table.loc[n_state, :].max()

        self.q_table.loc[c_state, action] += self.lr * (q_target - q_predict)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


if __name__ == '__main__':
    qtable = QLearningTable(actions=list(range(3)))
    print(qtable.choose_action("abc"))
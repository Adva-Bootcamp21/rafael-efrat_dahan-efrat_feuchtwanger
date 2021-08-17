import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/train.csv", index_col=[0], usecols=lambda x: x != 'targetName')
print(data)


def calc_distance(row):
    num_not_null = sum(row.notnull().sum().values)
    return np.linalg.norm(row.iloc[:, num_not_null-7:num_not_null-5].values - row.iloc[:, 0:2].values)

def calc_hist(group):
    x = []
    for i in range(len(group)):
        x.append(calc_distance(group.iloc[[i]]))
    plt.hist(x)
    plt.show()
    return


print(data.groupby(by="class").filter(lambda group: calc_hist(group)))


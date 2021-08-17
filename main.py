import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/train.csv", index_col=[0], usecols=lambda x: x != 'targetName')


print(data)


def calc_distance(row):
    num_not_null = sum(row.notnull().sum().values)
    return np.linalg.norm(row.iloc[:, num_not_null - 7:num_not_null - 5].values - row.iloc[:, 0:2].values)


def calc_hist(group):
    plt.hist([calc_distance(group.iloc[[i]]) for i in range(len(group))])
    plt.show()
    return


data.groupby(by="class").filter(lambda group: calc_hist(group))


def draw_route(row, color=None):
    num_not_null = sum(row.iloc[[0]].notnull().sum().values)
    xs = row.iloc[[0], 1:num_not_null:7].values.tolist()
    zs = row.iloc[[0], 3:num_not_null:7].values.tolist()
    plt.plot(xs[0], zs[0], color=color)


draw_route(data.iloc[[0]])
plt.show()


def draw_sum_route(group, color, num):
    for i in range(min(len(group), num)):
        draw_route(group.iloc[[i]], color)


def draw_sum_length_route(group, color, num):
    i, num_graph = 0, 0
    while num_graph < num and i < len(group):
        num_not_null = sum(group.iloc[[i]].notnull().sum().values)
        if num_not_null - 1 == 7 * 15:
            xs = group.iloc[[i], 1:num_not_null:7].values.tolist()
            zs = group.iloc[[i], 3:num_not_null:7].values.tolist()
            plt.plot(xs[0], zs[0], color=color)
            num_graph += 1
        i += 1


def draw_sum_length_route_up_and_down(group, color, num):
    index, num_graph = 0, 0
    while num_graph < num and index < len(group):
        num_not_null = sum(group.iloc[[index]].notnull().sum().values)
        if num_not_null - 1 == 7 * 15:
            velzs = group.iloc[[index], 6:num_not_null:7].values.tolist()
            down, up = False, False
            for i in range(0, len(velzs[0])):
                if velzs[0][i] < 0:
                    down = True
                elif velzs[0][i] > 0:
                    up = True
                if down and up:
                    xs = group.iloc[[index], 1:num_not_null:7].values.tolist()
                    zs = group.iloc[[index], 3:num_not_null:7].values.tolist()
                    plt.plot(xs[0], zs[0], color=color)
                    num_graph += 1
                    break
        index += 1


draw_sum_route(data.groupby(by="class").get_group(1), "red", 50)
draw_sum_route(data.groupby(by="class").get_group(6), "blue", 50)
plt.show()
draw_sum_length_route(data.groupby(by="class").get_group(1), "red", 50)
draw_sum_length_route(data.groupby(by="class").get_group(6), "blue", 50)
plt.show()
draw_sum_length_route_up_and_down(data.groupby(by="class").get_group(1), "red", 50)
draw_sum_length_route_up_and_down(data.groupby(by="class").get_group(6), "blue", 50)
plt.show()

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# q1
data = pd.read_csv("./data/train.csv", index_col=[0], usecols=lambda x: x != 'targetName')

print(data)


def calc_distance(row):
    num_not_null = sum(row.notnull().sum().values)
    return np.linalg.norm(row.iloc[:, num_not_null - 7:num_not_null - 5].values - row.iloc[:, 0:2].values)


def calc_hist(group):
    plt.hist([calc_distance(group.iloc[[i]]) for i in range(len(group))])
    plt.show()
    return


# data.groupby(by="class").filter(lambda group: calc_hist(group))

# q2
def draw_route(row, color=None):
    num_not_null = sum(row.iloc[[0]].notnull().sum().values)
    xs = row.iloc[[0], 1:num_not_null:7].values.tolist()
    zs = row.iloc[[0], 3:num_not_null:7].values.tolist()
    plt.plot(xs[0], zs[0], color=color)


# draw_route(data.iloc[[0]])
# plt.show()


def draw_sum_route(group, color, num):
    for i in range(min(len(group), num)):
        draw_route(group.iloc[[i]], color)


def draw_sum_length_route(group, color, num):
    i, num_graph = 0, 0
    while num_graph < num and i < len(group):
        num_not_null = sum(group.iloc[[i]].notnull().sum().values)
        if num_not_null - 1 == 7 * 30:
            xs = group.iloc[[i], 1:num_not_null:7].values.tolist()
            zs = group.iloc[[i], 3:num_not_null:7].values.tolist()
            plt.plot(xs[0], zs[0], color=color)
            num_graph += 1
        i += 1


def draw_sum_length_route_up_and_down(group, color, num):
    index, num_graph = 0, 0
    while num_graph < num and index < len(group):
        num_not_null = sum(group.iloc[[index]].notnull().sum().values)
        if num_not_null - 1 == 7 * 30:
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


# draw_sum_route(data.groupby(by="class").get_group(1), "red", 50)
# draw_sum_route(data.groupby(by="class").get_group(6), "blue", 50)
# plt.show()
# draw_sum_length_route(data.groupby(by="class").get_group(1), "red", 50)
# draw_sum_length_route(data.groupby(by="class").get_group(6), "blue", 50)
# plt.show()
# draw_sum_length_route_up_and_down(data.groupby(by="class").get_group(1), "red", 50)
# draw_sum_length_route_up_and_down(data.groupby(by="class").get_group(6), "blue", 50)
# plt.show()

# q3
data1 = data[(data["class"] == 1) + (data["class"] == 16)]
practice_data = data1.sample(frac=0.8)
test_data = data1[(data1.index.isin(practice_data.index))]
class_test = test_data.loc[:, ["class"]]
test_data = test_data.iloc[:, :-2]
print(practice_data)


# q4
def get_average_speed():
    # draw_sum_length_route_up_and_down(data.groupby(by="class").get_group(6), "blue", 50)
    # num_not_null = sum(row.iloc[[0]].notnull().sum().values)
    # x_speed = row.iloc[[0], 1:num_not_null:7].mean()
    # zs = row.iloc[[0], 3:num_not_null:7].values.tolist()
    # z_speed = sum(abs(zs))/len(zs)
    average_1_velx = practice_data.groupby(by="class").get_group(1).iloc[:, 4::7].mean(axis=1)
    average_16_velx = practice_data.groupby(by="class").get_group(16).iloc[:, 4::7].mean(axis=1)
    all_mean = test_data.iloc[:, 4::7].mean(axis=1)
    temp = (all_mean > (average_1_velx.mean() + average_16_velx.mean()) * 0.32)
    average_1_velz = abs(practice_data.groupby(by="class").get_group(1).iloc[:, 3::7]).mean(axis=1)
    average_16_velz = abs(practice_data.groupby(by="class").get_group(16).iloc[:, 3::7]).mean(axis=1)
    all_mean_z = abs(test_data.iloc[:, 3::7]).mean(axis=1)
    temp &= (all_mean_z > (average_1_velz.mean() + average_16_velz.mean()) * 0.32)
    print(temp)
    print(((practice_data["class"][temp] == 16).value_counts(True) + (practice_data["class"][~temp] == 1).value_counts(
        True)) / 2)


get_average_speed()

data1 = data1.fillna(0)
X = data1.iloc[:, 0:-1]
y = data1.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
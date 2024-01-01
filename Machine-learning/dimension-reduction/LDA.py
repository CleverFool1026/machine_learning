"""
线性判别分析LDA， 一种有监督的线性降维算法,
与PCA不同，LDA是为了降维后的数据尽可能地容易被区分，

在训练过程中，通过将训练样本投影到低维空间上，使得同类地投影点方差更小，不同类的更大
LDA更多的考虑了标注，希望投影后不同类别之间地数据点距离更大，同一类别地数据点更为紧凑
"""
import numpy as np
# 下面用鸢尾花数据集进行分析
from numpy import *
from numpy.linalg import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 导入鸢尾花数据集
print(load_iris().keys())
iris_data = load_iris()
# print(iris_data.data)                      # 4个属性，萼片长度，萼片宽度，花瓣长度，花瓣宽度
# print(iris_data.target)                   # 3个类别'setosa'-0 'versicolor'-1 'virginica'-2

data = pd.DataFrame(iris_data.data, columns=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度'])
label = pd.DataFrame(iris_data.target, columns=['类别'])
# print(label.head())

# 合并数据集
X = pd.concat([data, label], axis=1)
# print(X)

# 计算数据集的平均向量
mean_vector = []
class_label = np.unique(label)
n_class = class_label.shape[0]
X = np.array(X)
# print(X[X[:, -1] == 0][1:5, :])

for i in class_label:
    mean_vector.append(np.mean(X[X[:, -1] == i], axis=0))
mean_vector = np.array(mean_vector)
mean_vector = mean_vector[:, 0:4]
print(mean_vector)

# 计算类内散度矩阵
S_W = np.zeros((4, 4))
for i in class_label:
    S1 = np.zeros((4, 4))
    for row in X[X[:, -1] == i][:, 0:4]:
        row = row.reshape(4, 1)
        mv = mean_vector[i, :].reshape(4, 1)
        S1 += (row - mv).dot((row - mv).T)
    S_W += S1
print(S_W)

# 计算类间散度矩阵
S_B = np.zeros((4, 4))
mean_X = np.mean(X[:, 0:4], axis=0)
# for i in class_label:
#     S2 = np.zeros((4, 4))
#     N = X[X[:, -1] == i].shape[0]
#     for row in X[X[:, -1] == i][:, 0:4]:
#         row = row.reshape(4, 1)
#         mv = mean_vector[i, :].reshape(4, 1)
#         S2 += (row - mv).dot((row - mv).T)
#     S_B += S2*N
# print(S_B)
for i, mean_vec in enumerate(mean_vector):
    N = X[X[:, -1] == i].shape[0]
    mean_vec = mean_vec.reshape(4, 1)
    mean_X = mean_X.reshape(4, 1)
    S_B += N*(mean_vec - mean_X).dot((mean_vec - mean_X).T)
print(S_B)

# 计算特征值和特征向量
eig_values, eig_vector = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# print(eig_values)
# print(eig_vector)

# 用最大的特征值对应的特征向量做投射
idx = np.argsort(eig_values)
# 提取第一个
k = 1
W = eig_vector[idx[: -k-1: -1], :]
y = np.dot(X[:, 0:4], W.T)
y = y.real
print(y)
marks = ['^', '*', 'o']
colors = ['r', 'y', 'g']

for i in range(len(X[:, -1])):
    plt.scatter(y[i,:], y[i, :], c = colors[int(X[i, -1])], marker = marks[int(X[i, -1])])

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.title('降维后分类图')
plt.show()


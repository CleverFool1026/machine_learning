# 主成分分析是最常用的线性降维手段
"""
现在基于sklearn和Numpy库随机生成2个类别共40个三维空间的样本点进行说明
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成两个样本
mean1 = np.array([0, 0, 0])
cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class1 = np.random.multivariate_normal(mean1, cov1, 20).T

mean2 = np.array([1, 1, 1])
cov2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2 = np.random.multivariate_normal(mean2, cov2, 20).T

all_samples = np.concatenate((class1, class2), axis=1)
mean_x = np.mean(all_samples[0, :])
mean_y = np.mean(all_samples[1, :])
mean_z = np.mean(all_samples[2, :])

# 计算样本离差阵S，除此以外也可以用协差阵求解（numpy.cov()方法）
scatter_matrix = np.zeros((3, 3))
mean_vector = [[mean_x], [mean_y], [mean_z]]
for i in range(all_samples.shape[1]):
    scatter_matrix += ((all_samples[:, i].reshape(3, 1) - mean_vector).
                       dot((all_samples[:, i].reshape(3, 1) - mean_vector).T))

print(scatter_matrix)

# 计算特征值和特征向量并排序，提出前两个特征值-特征向量对作为坐标
# 将三维投射到二维空间中
# 2X3的特征向量矩阵W
eig_value, eig_vec = np.linalg.eig(scatter_matrix)
idx = np.argsort(eig_value)
# 提取前两个
k = 2
W = eig_vec[idx[: -k-1: -1], :]

print('特征值：', eig_value)
print('特征向量：', eig_vec)
print(W)

# y = W'*x
y = np.dot(W, all_samples)
y = np.array(y)
print(y.shape)

# 画出三维散点图
# x = all_samples[0, :]
# y = all_samples[1, :]
# z = all_samples[2, :]
# fig = plt.figure()
# ax1 = Axes3D(fig)
# fig.add_axes(ax1)
# ax1.scatter(class1[0, :], class1[1, :], class1[2, :], c = 'b')
# ax1.scatter(class2[0, :], class2[1, :], class2[2, :], c = 'r')
# ax1.set_label('x')
# ax1.set_label('y')
# ax1.set_label('z')
# plt.show()

# 画出降维后的二维图
ax2 = plt.subplots(figsize = (10, 10))
x1 = y[0, 0:20]
y1 = y[1, 0:20]
x2 = y[0, 21:40]
y2 = y[1, 21:40]
plt.scatter(x1, y1, c = 'g', marker='^')
plt.scatter(x2, y2, c = 'r', marker='^')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('降维后')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.show()


"""
LLE是一种典型的非线性降维算法，
这一算法要求每个数据点都可以由近邻点的线性加权组合构造得到，
从而使降维后的数据也能基本保持原有流形结构
"""

# 用LLE对“瑞士卷”数据集进行降维
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, color = datasets._samples_generator.make_swiss_roll(n_samples=1500)
x_r, err = manifold.locally_linear_embedding(x, n_neighbors=10, n_components=2) # 近邻数为10，降到二维

# 可视化
fig = plt.figure()
ax1 = Axes3D(fig)
fig.add_axes(ax1)
ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c = color, cmap='Blues')     # 原始数据
plt.show()

fig = plt.figure()
plt.scatter(x_r[:, 0], x_r[:, 1], c = color, cmap='Blues')      # 投射后
plt.show()


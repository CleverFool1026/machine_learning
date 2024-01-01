"""
拉普拉斯特征映射解决问题思路与LLE类似，是一种基于图的降维算法，
使相互关联的点在降维后的空间中尽可能地靠近
"""
# 用LE对“瑞士卷”数据集进行降维
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, color = datasets._samples_generator.make_swiss_roll(n_samples=1500)

# 可视化
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121, projection = '3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c = color, cmap='Oranges_r')
ax.view_init(4, -72)

se = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)         # 近邻数为10，降到二维
y = se.fit_transform(x)
# 降维后可视化
ax2 = fig.add_subplot(122)
plt.scatter(y[:, 0], y[:, 1], c = color, cmap='Oranges_r')

plt.show()


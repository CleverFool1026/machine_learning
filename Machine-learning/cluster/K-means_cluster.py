"""
基于划分的聚类是一种简单、常用的聚类
下面介绍k-means聚类和k-prototype聚类算法

对于k-means聚类，其优点是速度快、易于理解。缺点是容易局部收敛，在大规模的数据集上求解速度较慢
且对于离群点和噪声点非常敏感；初始中心点的选取对结果的影响很大
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
plt.rcParams['font.sans-serif'] = ['SimHei']        # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False        # 用来正常显示符号

np.random.seed(5)
iris = datasets.load_iris()
x = iris.data
y = iris.target
est = KMeans(n_clusters=3, n_init=10)
est.fit(x)
labels = est.labels_

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
fig.add_axes(ax)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], c = labels.astype(np.float_), edgecolors='k')
ax.set_xlabel('花瓣宽度')
ax.set_ylabel('萼片长度')
ax.set_zlabel('花瓣长度')
ax.set_title('3类')
ax._dist = 10
# plt.savefig('D:/work pythonproject/Picture_set/鸢尾花分类.png')
plt.show()


"""
k-medoids算法的迭代过程与k-means类似，主要差别在于聚类（簇）中心的选择方法，
可见，k-medoids算法对噪声的稳健性比较好，但其速度比较慢，不太适合数据量大的样本聚类
围绕中心点划分（Partitioning Around Medoids， PAM）算法是k-medoids聚类的一种流行的实现。

k-prototype算法加入了描述数据簇的原型和混合数据之间的相异度的计算公式，能快速处理混合类型数据集的聚类问题
"""





"""
从PCA的原理上看这种方法并没有改变各样本之间的关系
只是运用了新的坐标系，PCA的主要缺点是，当数据量和数据维度非常大的时候
用协差阵的方法解PCA就会变得非常低微， 解决办法是采用奇异值分解——singlear value decomposition,SVD
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
print(data.head())
print(data.info())

"""
利用‘data’中的数据作为linalg.svd()方法的输入，
得到左奇异矩阵U，奇异值矩阵S，右奇异值矩阵V，
选择U中前两个特征分别作为二维平面的x, y坐标进行可视化
"""
sample, features = data.shape
# print(sample, features)
U, S, V = linalg.svd(data)

newdata = U[:, :2]
# print(newdata)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
marks = ['o', '^', '+']
colors = ['r', 'y', 'g']
for i in range(sample):
    ax.scatter(newdata[i, 0], newdata[i, 1], c=colors[int(iris_data.target[i])], marker=marks[int(iris_data.target[i])])

plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.show()



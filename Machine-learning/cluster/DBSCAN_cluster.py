"""
基于密度的聚类
基于领域来描述样本的密度，具体过程参见网上教程或教材
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4] # #表示我们只取特征空间中的4个维度

estimator = DBSCAN(eps=0.4,min_samples=9) # 构造聚类器
estimator.fit(X) # 聚类
label_pred2 = estimator.labels_ # 获取聚类标签


# 绘制结果
x0 = X[label_pred2 == 0]
x1 = X[label_pred2 == 1]
x2 = X[label_pred2 == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()





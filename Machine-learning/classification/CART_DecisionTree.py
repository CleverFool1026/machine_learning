"""
CART算法是构建决策树的一种常用算法，
下面是基于sklearn库的CART算法的示例代码。
通过构建决策树（采用 Gini作为指标）对随机生成的数字进行分类
"""
import numpy as np
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randint(10, size=(100, 4))
Y = np.random.randint(2, size=100)
a = np.column_stack((Y, X))
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
clf = clf.fit(X, Y)  # 这里改为了 clf.fit(X, Y)

# dot_clf = tree.export_graphviz(clf)
# graph = pydotplus.graph_from_dot_data(dot_clf)
# graph.write_png('CART_Tree.png')
plt.figure(figsize=(12, 8))
plt.title('CART_Tree')
tree.plot_tree(clf)
# plt.savefig('D:/work pythonproject/Picture_set/CART_Tree.png')
plt.show()


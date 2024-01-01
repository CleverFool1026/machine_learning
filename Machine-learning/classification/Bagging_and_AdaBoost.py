"""
Ensemble learning——集成学习
装袋法（bagging）又称引导聚集法，其原理是通过组合多个训练集的分类结果来提升分类效果
假设有一个大小为n的训练样本集S，袋装法是从S中多次放回采样去除大小为n'(n'<n)的m个训练集,
对于每个训练集Si，均选用特定的学习方法，建立分类模型。
对于新的测试样本，所建立的m个分类模型将返回m个预测分类结果，
袋装法构建的模型最终返回的是这m个预测结果中占多数的分类结果。
袋装法由于多次采样，每个样本被选中的概率相同，噪声数据的影响减小，因此不太容易受到过拟合的影响。
下面利用鸢尾花数据集进行具体说明
"""
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target
# 分类器及交叉验证
seed = 42
kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier(criterion='gini', max_depth=2)
cart = cart.fit(x, y)
result = cross_val_score(cart, x, y, cv=kfold)
print('CART树的结果：', result.mean())
model = BaggingClassifier(estimator=cart, n_estimators=100, random_state=seed)
result = cross_val_score(model, x, y, cv=kfold)
print('袋装法提升结果：', result.mean())
"""
CART树的结果： 0.9333333333333333
袋装法提升结果： 0.9466666666666667
可以看到袋装法对模型效果有一定提升，当然，提升程度与原模型的结构和数据质量有关，
如果CART的深度设置为3或者5，原算法本身的效果就会比较好，袋装法可能就没有提升空间。
"""

"""
提升法与袋装法类似，但引入了权重，下面用肺癌数据集进行说明
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

data = datasets.load_breast_cancer()
x2 = data.data
y2 = data.target

seed2 = 42
kfold2 = KFold(n_splits=10)
cart2 = DecisionTreeClassifier(criterion='gini', max_depth=3)
cart2 = cart2.fit(x2, y2)
result2 = cross_val_score(cart2, x2, y2, cv=kfold2)
print('CART树的结果：', result2.mean())
model2 = AdaBoostClassifier(estimator=cart2, n_estimators=100, random_state=seed2)  # n_estimators=100创建100个分类模型
result2 = cross_val_score(model2, x2, y2, cv=kfold2)
print('提升法改进结果：', result2.mean())

"""
CART树的结果： 0.9314536340852131
提升法改进结果： 0.9718984962406015
"""
# 可以看到提升法对当前决策树分类器的分类效果改进较大

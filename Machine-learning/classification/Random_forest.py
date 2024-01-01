"""
随机森林是专为决策树分类器设计的一种集成方式，是袋装法的一种拓展
随机森林算法每次从所有属性中抽取f个属性，然后从f个属性中选取一个最优的属性作为其分支属性
这样使整个模型的随机性更强，从而使模型的泛化能力更强
当f等于属性总数(M)时，就变成了袋装法的集成方式，通常f的取值为小于log_2(M+1)的最大整数
而随机森林算法使用弱分类决策树通常为CART算法，随机森林算法思路简单，易实现，却有着较好的分类效果
下面具体说明
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 随机生成测试样本
x, y = make_blobs(n_samples=1000, n_features=6, centers=50, random_state=0)
plt.scatter(x[:, 0], x[:, 1], c = y)
# plt.show()

# 构造决策树分类模型，随机森林分类模型
# 叶节点个数最少为2
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
score = cross_val_score(clf, x, y)
print('决策树结果：', score.mean())
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
score = cross_val_score(clf, x, y)
print('随机森林结果：', score.mean())

# 极限随机森林
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
score = cross_val_score(clf, x, y)
print('极限森林结果：', score.mean())

"""
决策树结果： 0.9490000000000001
随机森林结果： 0.9950000000000001
极限森林结果： 0.9970000000000001
"""
# 决策树<随机森林<极限随机森林


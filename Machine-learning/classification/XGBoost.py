"""
梯度提升决策树（GBDT）是一种迭代决策树算法，主要用于回归，改进后也可用于实现分类的任务
而XGBoost是具体实现GBDT算法最为出色的提升系统，相关原理较为复杂，暂不详述
下面是一个简单的例子
"""

import pandas as pd
import xgboost as xgb

df = pd.DataFrame({'x':[1, 2, 3], 'y':[10, 20, 30]})
x_train = df.drop('y', axis=1)
y_train = df['y']
xgb_train = xgb.DMatrix(x_train, y_train)
params = {'objective':'reg:squarederror', 'booster':'gblinear'}
# objective代表学习目标，reg:squarederror代表使用平方误差的方法，
# booster代表控制每一步提升的方式，这里使用线性模型gblinear,
# 也可以选择基于树的模型tree
gbm = xgb.train(dtrain = xgb_train, params = params)        # 训练模型，关键一步
y_predict = gbm.predict(xgb.DMatrix(pd.DataFrame({'x':[4, 5]})))        # 预测4，5的y值
print(y_predict)

"""
[29.495766 33.612526]
"""

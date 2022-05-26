# -*- coding:utf-8 -*-
"""
作者：yzf93
日期:2022年05月06日
"""
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt

# 西瓜数据集3.0α
data_set = {"data": np.array([[0.697, 0.460],
                              [0.774, 0.376],
                              [0.634, 0.264],
                              [0.608, 0.318],
                              [0.556, 0.215],
                              [0.403, 0.237],
                              [0.481, 0.149],
                              [0.437, 0.211],
                              [0.666, 0.091],
                              [0.243, 0.267],
                              [0.245, 0.057],
                              [0.343, 0.099],
                              [0.639, 0.161],
                              [0.657, 0.198],
                              [0.360, 0.370],
                              [0.593, 0.042],
                              [0.719, 0.103]]), "target": np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])}

data = data_set['data']
target = data_set['target']

# sklearn中adaboost默认的基分类器就是深度为1的决策树
boost1 = AdaBoostClassifier(n_estimators=15)  # 集成规模为3
boost2 = AdaBoostClassifier(n_estimators=50)  # 集成规模为5
boost3 = AdaBoostClassifier(n_estimators=100)  # 集成规模为11

# 训练模型
boost1.fit(data, target)
boost2.fit(data, target)
boost3.fit(data, target)

# 训练结果可视化
xMin, yMin = 0, 0  # 坐标轴最小值
xMax, yMax = 0.8, 0.6  # 坐标轴最大值
# 将x中每一个数据和y中每一个数据组合生成很多点,然后将这些点的x坐标放入到X中,y坐标放入Y中
xSet, ySet = np.meshgrid(np.arange(xMin, xMax, 0.02), np.arange(yMin, yMax, 0.02))
label_set = []
for item in [boost1, boost2, boost3]:
    # 按列叠加两个矩阵
    label = item.predict(np.c_[xSet.ravel(), ySet.ravel()]).reshape(xSet.shape)
    label_set.append(label)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
(ax0, ax1, ax2) = axes.flatten()
# k为索引，axi为第i张图的坐标轴
for k, ax in enumerate((ax0, ax1, ax2)):
    ax.contourf(xSet, ySet, label_set[k], cmap=plt.cm.Set1)
    # 依次遍历[0, 1], ['bad', 'good'], ['green', 'white']三个列表中索引相同的元素
    for i, n, c in zip([0, 1], ['bad', 'good'], ['green', 'white']):
        idx = np.where(target == i)
        ax.scatter(data[idx, 0], data[idx, 1], c=c, label=n)
    # 开始画图
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper left')
    ax.set_ylabel('sugar')
    ax.set_xlabel('densty')
    ax.set_title('decision boundary for %s' % (k + 1))
plt.show()

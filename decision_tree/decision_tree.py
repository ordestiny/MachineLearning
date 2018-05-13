# ID3/C4.5算法

import math
import numpy as np
import operator
from tree_visual import *
from  decision_utils import *

"""
生成决策树
"""


# 生成一些测试数据
def generate_data(num=100):
    return np.random.randint(num, size=num)

# 导入数据
def createDataSet():
    dataSet = [['youth', 'no', 'no', 1, 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', 1, 'agree'],
               ['youth', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', 1, 'refuse'],
               ]
    labels = ["age", "working", "house", "credit_situation"]
    return dataSet, labels


# 测试代码
if __name__ == "__main__":
    # 用于生成决策树
    myDat, labels = createDataSet()
    myDat2, labels2 = createDataSet()
    print(labels)
    # 生成决策树
    myTree = creatTree(myDat, labels)
    # print(myTree)
    # print(labels)
    # 可视化
    createPlot(myTree)
    # 预测
    classLabel = classify(myTree, labels2, ['youth', 'no', 'no', 1])
    print("class:"+ classLabel)

    loss = calcTestErr(myTree,myDat,labels2)
    print("loss:"+ str(loss))


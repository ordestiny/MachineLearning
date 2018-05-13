import numpy as np
import math

def classUnique(lst):
    return dict(zip(*np.unique(lst, return_counts=True)))

# 计算香农熵（表示变量不确定性的度量）
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCnt = classUnique([x[-1] for x in dataSet]) # 统计各个类别的频数
    shannonEnt = 0.0
    for key in labelCnt:
        prob  =  float(labelCnt[key])/numEntries
        shannonEnt -=  prob * math.log(prob,2)
    return shannonEnt

# def calcShannonEnt(dataSet):
#     numEntries = len(dataSet)
#     labelCounts = {}##创建字典
#     for featVec in dataSet:
#         currentLabel = featVec[-1]
#         if currentLabel not in labelCounts.keys():
#             labelCounts[currentLabel] = 0
#         labelCounts[currentLabel] += 1
#     shannonEnt = 0.0
#     for key in labelCounts:
#         prob = float(labelCounts[key]) / numEntries
#         shannonEnt -= prob * math.log(prob,2)
#     return shannonEnt

# 计算列表各个元素的频数
def classUnique(lst):
    return dict(zip(*np.unique(lst, return_counts=True)))

# 计算条件熵（已知随机变量X的条件下，随机变量Y的不确定性）
def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    """
    :param dataSet:
    :param i: 维度
    :param featList:特征列表
    :param uniqueVals: 数据集特征集合
    :return:
    """
    conditionalEnt = 0.0
    for value in uniqueVals:
        subDateSet = splitDateSet(dataSet,i,value)
        # 子集概率
        prob = len(subDateSet)/ float(len(dataSet)) # 极大似然估计概率
        # 子集的熵，和为该条件下的条件熵
        conditionalEnt += prob * calcShannonEnt(subDateSet)
    return conditionalEnt

# 按照给定的特征，划分数据集
def splitDateSet(dataSet,axis,value):
    retDateSet = []
    for featVec in dataSet: # 遍历所有的样本
        if featVec[axis] == value: # 若某一个特征值与指定的特征值相同，将该特征值去掉，剩下的特征值返回
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDateSet.append(reducedFeatVec)
    return retDateSet

# 计算信息增益（表示得知特征X的信息而使得Y的信息不确定减少的程度）
def calcInformationGain(dataSet,baseEntropy,i):
    """
    :param dataSet:
    :param baseEntropy: 数据集的信息熵
    :param i: 特征维度
    :return: 特征i对数据集的信息增益g(D｜x_i)
    """
    featList = [example[i] for example in dataSet] #  取得当前特征对应列下的值
    uniqueVals = set(featList) # 特征值去重
    newEntropy  = calcConditionalEntropy(dataSet,i,featList,uniqueVals)
    infoGain = baseEntropy -  newEntropy # 信息增益
    return infoGain

#计算信息增益比（用信息增益比来选择特征的算法称为C4.5算法）
def calcInformationGainRatio(dataSet,baseEntropy,i):
    return calcInformationGain(dataSet,baseEntropy,i) / baseEntropy

# 计算基尼指数
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = classUnique([x[-1] for x in dataSet]) # 统计各个类别的频数
    Gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Gini -= prob * prob
    return Gini

# 计算给定特征下的基尼指数
def calcGiniWithFeat(dataSet, feature, value):
    D0 = []; D1 = []
    # 根据特征划分数据
    for featVec in dataSet:
        if featVec[feature] == value:
            D0.append(featVec)
        else:
            D1.append(featVec)
    Gini = len(D0) / len(dataSet) * calcGini(D0) + len(D1) / len(dataSet) * calcGini(D1)
    return Gini

# 多数表决的方法来决定该叶子节点的分类：即叶节点的所属类别，是该叶节点中样本数最多的一类
def majorityCnt(classList):
    classCount={}
    for vote in classList:                  # 统计所有类标签的频数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) # 排序
    return sortedClassCount[0][0] # 返回数量最多的类型


"""
特征选择
"""
# ID3算法：选择信息熵最大的特征，即最好的特征，返回下标
def chooseBestFeatureToSplitByID3(dataSet):
    numFeature = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        infoGain =  calcInformationGain(dataSet,baseEntropy,i)
        if(infoGain > baseInfoGain):
            baseInfoGain = infoGain
            bestFeature = i
    return  bestFeature

# C4.5算法:用信息增益率来选择属性
def chooseBestFeatureToSplitByC45(dataSet):
    '''
            选择最好的数据集划分方式
    :param dataSet:
    :return: 划分结果
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列yes分类标签，不属于特征变量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRatio(dataSet, baseEntropy, i)    # 计算信息增益比
        if (infoGainRate > bestInfoGainRate):  # 选择最大的信息增益比
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度

def chooseBestFeatureToSplitByCART(dataSet):
    numFeatures = len(dataSet[0])-1
    bestGini = 0
    bestFeat = 0
    bestValue = 0
    newGini = 0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        for splitVal in uniqueVals:
            newGini = calcGiniWithFeat(dataSet, i, splitVal)
            if newGini < bestGini:
                bestFeat = i
                bestGini = newGini
    return bestFeat


"""
决策树的生成，是一个递归过程。

1、计算所有的特征（label），对于数据集（dataSet）的信息增益，选取信息增益最大的特征 A，作为tree的第一个节点

2、遍历上一步得到的特征 A 在数据集中的所有取值(如A1，A2...), 以A = A_i 取 dataSet 子集，该子集排除A特征，每个子集。
   以上一步的方法，取出最大增益的特征为作为下一级子节点，持续递归。

3、递归至遍历完所有的子集结束，将树输出。
"""
def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # 标签值
    # 类别完全相同，即只有一个类，返回类标签即可。
    if classList.count(classList[0]) ==  len(classList):
        return  classList[0]

    # 若没有特征值，返回数据集中类别数量最多的类别
    if len(dataSet[0]) == -1:
        return  majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplitByCART(dataSet)
    bestFeatLabel  = labels[bestFeat]

    tree  = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 删除已选的选择的特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals  = set(featValues) # set用于创建无序不重复元素集，相当于去重
    for value in uniqueVals:
        # 除最佳特征外的其他特征
        subLabels  = labels[:]
        # splitDateSet方法，将最佳特征的数据集去掉，然后递归，从剩下的特征中，重新找最好的特征，结果做为树的子集。
        tree[bestFeatLabel][value] = creatTree(splitDateSet(dataSet,bestFeat,value),subLabels)
    return tree

"""
执行分类
"""
def classify(inputTree,featLabels,testVec):
    '''
    利用决策树进行分类
    :param: inputTree:构造好的决策树模型
    :param: featLabels:所有的类标签
    :param: testVec:测试数据
    :return: 分类决策结果
    '''
    # firstStr = inputTree.keys()[0]
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    print(firstStr)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def calcTestErr(myTree,testData,labels):
    errorCount = 0.0
    for i in range(len(testData)):
        if classify(myTree,labels,testData[i]) != testData[i][-1]:
            errorCount += 1
    return float(errorCount)
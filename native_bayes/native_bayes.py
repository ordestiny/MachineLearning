#coding:utf-8
# 极大似然估计  朴素贝叶斯算法
import pandas as pd
import numpy as np

"""
bayes分类的原理：

计算输入的测试样本 X，是各个分类 Y 的可能性，即概率 P(Y｜X)

计算方法：

1、 计算各个分类的概率，P(Y1),P(Y2),...,P(Yn)
2、 计算从各个分类中，出现X的概率，P(X｜Y1),P(X｜Y2),....,P(X｜Yn)，其中P(X|Y1) = P(XY1)/P(Y1)
3、 计算出现 X 的概率，即P(X) = P(X｜Y1) * P(Y1) + P(X｜Y2) * P(Y2)+...+P(X｜Yn) * P(Yn)

样本X 属于Yn类的概率为：P(Yn｜X) = P(X｜Yn) * P(Yn) / P(X)，取概率最大的分类做为输出。
"""

class NaiveBayes(object):
    def getTrainSet(self):
        dataSet = pd.read_csv('naivebayes_data.csv')
        dataSetNP = np.array(dataSet)  #将数据由dataframe类型转换为数组类型
        trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #训练数据x1,x2
        labels = dataSetNP[:,dataSetNP.shape[1]-1]        #训练数据所对应的所属类型Y
        return trainData, labels

    def classify(self, trainData, labels, features):
        # 求labels中每个label的先验概率，即P(Y)
        labels = list(labels)    #转换为list类型
        P_y = {}       #存入label的概率
        for label in labels:
            P_y[label] = labels.count(label)/float(len(labels))   # p = count(y) / count(Y)

        # 求label与feature同时发生的概率，即交集概率P(XY)
        # 计算得到如:P(X1Y1),P(X2Y1),P(X1Y2),P(X2Y2),....
        P_xy = {}
        for y in P_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]   # labels中出现y值的所有数值的下标索引
            for j in range(len(features)):
                # features[j] 在trainData[:,j]中出现的值的所有下标索引
                x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
                xy_count = len(set(x_index) & set(y_index))     # set(x_index)&set(y_index)列出两个表相同的元素
                pkey = str(features[j]) + '*' + str(y)
                P_xy[pkey] = xy_count / float(len(labels))

        # 求条件概率P(Xn|Yn) = P(XnYn)/P(Yn)
        P = {}
        for y in P_y.keys():
            for x in features:
                pkey = str(x) + '|' + str(y)
                P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])

        # 求[2,'S']所属类别
        F = {}   #[2,'S']属于各个类别的概率
        for y in P_y:
            F[y] = P_y[y]
            for x in features:
                F[y] = F[y]*P[str(x)+'|'+str(y)]     #P[y|X] = P[X|y]*P[y]/P[X]，分母相等，即求分子最大即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]

        features_label = max(F, key=F.get)  #概率最大值对应的类别
        return features_label

# 利用矩阵运算
class NaiveBayes_2(object):
    def getTrainSet(self):
        dataSet = pd.read_csv('naivebayes_data.csv',header=0)
        trainData = dataSet[:,0:-1]
        labels = dataSet.iloc[:,-1]
        return trainData, labels

    # 各个分类出现的概率P(Y)
    def getPY(self,labels):
        return labels.value_counts() / labels.count()

if __name__ == '__main__':
    nb = NaiveBayes()
    # 训练数据
    trainData, labels = nb.getTrainSet()
    # x1,x2
    features = [2,'S']
    # 该特征应属于哪一类
    result = nb.classify(trainData, labels, features)
    print(features,'属于',result)
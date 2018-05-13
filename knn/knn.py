#coding:utf-8
import pandas as pd
import numpy as np

class KNNa(object):

    #获取训练数据集
    def getTrainData(self):
        dataSet = pd.read_csv('iris.csv', header=None)
        dataSetNP = np.array(dataSet[1:150])
        trainData = dataSetNP[:,0:dataSetNP.shape[1]-1]   #获得训练数据
        labels = dataSetNP[:,dataSetNP.shape[1]-1]    #获得训练数据类别
        return trainData,labels
    #得到测试数据的类别
    def classify(self, testData, trainData, labels, k):
        #计算测试数据与训练数据之间的欧式距离
        dist = []
        for i in range(len(trainData)):
            td = trainData[i,:]   #训练数据
            dist.append(np.linalg.norm(testData - td))   #欧式距离 （平方和，再开方）
        dist_collection = np.array(dist)   #获得所有的欧氏距离，并转换为array类型
        dist_index = dist_collection.argsort()[0:k]   #按升序排列，获得前k个下标
        k_labels = labels[dist_index]   #获得对应下标的类别

        #计算k个数据中，类别的数目
        k_labels = list(k_labels)   #转换为list类型
        labels_count = {}
        for i in k_labels:
            labels_count[i] = k_labels.count(i)  #计算每个类别出现的次数
        testData_label = max(labels_count, key=labels_count.get)   #次数出现最多的类别
        return testData_label


if __name__ == '__main__':
    kn = KNNa()
    trainData,labels = kn.getTrainData()   #获得训练数据集,iris从第2行到第150行的149条数据
    testData = np.array([5.1, 3.5, 1.4, 0.2])   #取iris中的数据的第1行
    k = 10   #最近邻数据数目
    testData_label = kn.classify(testData,trainData,labels,k)    #获得测试数据的分类类别
    print('测试数据的类别：',testData_label)
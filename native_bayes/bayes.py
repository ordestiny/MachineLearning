import numpy as np
import copy as cp

#获得特征向量可能值
def createWordList(data):
    wordSet = set([])
    for document in data:
        wordSet = wordSet | set(document)
    return list(wordSet)

#将多维数据转化为一维向量，方便计算
def word2Vec(wordList,inputWord):
    returnVec = [0]*len(wordList)
    for word in inputWord:
        if word in wordList:
            returnVec[wordList.index(word)] = 1
    return returnVec

#训练函数，根据给定数据和标签，计算概率
def train(trainMatrix,trainLabels):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = (sum(trainLabels)+1.0)/(float(numTrainDocs)+2.0*1.0)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 3.0 + len(trainLabels)-sum(trainLabels)
    p1Denom = 3.0 + sum(trainLabels)
    for i in range(numTrainDocs):
        if trainLabels[i] == 1:
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
    p0Vect = np.log(p0Num/p0Denom)
    p1Vect = np.log(p1Num/p1Denom)

    return p0Vect,p1Vect,pAbusive

#分类函数
def classify(vec2Clssify,p0Vect,p1Vect,pClass1):
    p1 = sum(vec2Clssify*p1Vect) + np.log(pClass1)
    p0 = sum(vec2Clssify*p0Vect) + np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def main():
    data = [[1,'s'],[1,'m'],[1,'m'],[1,'s'],[1,'s'],[2,'s'],[2,'m'],[2,'m'],[2,'l'],[2,'l'],[3,'l'],[3,'m'],[3,'m'],[3,'l'],[3,'l']]
    labels = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
    wordList = createWordList(data)
    dataMatrix = []
    for item in data:
        dataMatrix.append(word2Vec(wordList,item))
    p0,p1,pAB = train(dataMatrix,labels)
    goal = [3,'l']
    wordVec = np.array(word2Vec(wordList,goal))
    print(classify(wordVec,p0,p1,pAB))

if __name__ == '__main__':
    main()
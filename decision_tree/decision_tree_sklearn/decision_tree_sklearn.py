import numpy as np
import scipy as sp
from sklearn import  tree
from sklearn.metrics import  precision_recall_curve,classification_report
from sklearn.cross_validation import train_test_split

data =[]
labels =[]

with open("data.txt") as infile:
    for line in infile:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)

# 标签转换为0/1
y[labels =="fat"] =1

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# 使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion="entropy")
# print(clf)
clf.fit(x_train,y_train)

#把决策树结构写入文件
with open("tree.dot","w") as file:
    file = tree.export_graphviz(clf,out_file=file)

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print("每个特征的影响力:",clf.feature_importances_)

answer = clf.predict(x_train)
# 测试结果的打印
# print(x_train)
# print(answer)
# print(y_train)
print("mean:",np.mean(answer == y_train))
#准确率与召回率
prrecision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
answer  = clf.predict_proba(x)[:,1]  # predict_proba 返回该样本可能是各个类型的概率,predict返回所属类别（相当于取predict_proba中的概率最大值的类型返回）
print(classification_report(y,answer,target_names=["thin","fat"]))

from sklearn.datasets import load_files
import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

movie_reviews=load_files('dataset/cs_cornell_edu_com')

'''
上面的数据是通过load files来获取已经存放在本地的数据文件，是几百个对电影的英文评论，我们把它分成顶和踩
'''
#save
# sp.save('dataset/cs_cornell_edu_com/movie_data.npy',movie_reviews.data)
# sp.save('dataset/cs_cornell_edu_com/movie_target.npy',movie_reviews.target)

#read
movie_data=sp.load('dataset/cs_cornell_edu_com/movie_data.npy')
movie_target=sp.load('dataset/cs_cornell_edu_com/movie_target.npy')


'''
TfidfVectorizer 可以把一大堆文档转换成TF-IDF特征的矩阵。TF-IDF = TF*IDF

TF: Term Frequency, 用于衡量一个词在一个文件中的出现频率。
因为每个文档的长度的差别可以很大，因而一个词在某个文档中出现的次数可能远远大于另一个文档，所以词频通常就是一个词出现的次数除以文档的总长度，相当于是做了一次归一化。
TF(t) = (词t在文档中出现的总次数) / (文档的词总数).
比如文档“I love this movie”，单词“love”的TF为1/4。如果去掉停用词“I"和”this“，则为1/2。


IDF：逆向文件频率，用于衡量一个词的重要性。
计算词频TF的时候，所有的词语都被当做一样重要的，但是某些词，比如”is”, “of”, “that”很可能出现很多很多次,但是可能根本并不重要，因此我们需要减轻在多个文档中都频繁出现的词的权重。 
ID(t) = log_e(总文档数/词t出现的文档数)
IDF是为了凸显那种出现的少，但是占有强烈感情色彩的词语。比如“movie”这样的词的IDF=ln(12/5)=0.88，远小于“love”的IDF=ln(12/1)=2.48。

stop_words = 'english'，表示使用默认的英文停用词。可以使用count_vec.get_stop_words()查看TfidfVectorizer内置的所有停用词。
当然，在这里可以传递你自己的停用词list（比如这里的“movie”）
注意这些数据集可能存在非法字符问题。所以在构造count_vec时，传入了decode_error = 'ignore'，以忽略这些非法字符。
count_vec构造时默认传递了max_df=1，因此TF-IDF都做了规格化处理，以便将所有值约束在[0,1]之间。
'''
# BOOL型特征下的向量空间模型，注意，训练样本调用的是fit_transform接口，测试样本调用的是transform接口
count_vec = TfidfVectorizer(binary=False, decode_error='ignore',stop_words='english')

# 加载数据集，切分数据集80%训练，20%测试
x_train, x_test, y_train, y_test = train_test_split(movie_data, movie_target, test_size=0.2)
# 特征提取
x_train = count_vec.fit_transform(x_train)
x_test = count_vec.transform(x_test)

x=count_vec.transform(movie_data)
y=movie_target

# 调用MultinomialNB分类器
clf = MultinomialNB().fit(x_train, y_train)
doc_class_predicted = clf.predict(x_test)

print(count_vec.get_feature_names()) #这是一个获取相应的特征值的
print(u'训练集\n',x_train.toarray())
print(u'预测的结果\n',doc_class_predicted)
print(u'实际值\n',y)
print(u'平均成功概率\n',np.mean(doc_class_predicted==y_test))

# 准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
answer = clf.predict_proba(x_test)[:, 1]
report = answer > 0.5
print(classification_report(y_test, report, target_names=['neg', 'pos']))



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import scipy.optimize as scop

def readDate():
	header = ['user_id', 'item_id', 'rating', 'timestamp']
	df = pd.read_csv('movielens/ml-1m/ratings.dat', sep='::', names=header)
	return df

def getNumberOfUsers(df):
	return df.user_id.max()

def getNumberOfVideos(df):
	return df.item_id.max()

# 将数据集分割成测试和训练
def getTrainAndTestDate(df,testscale):
	train_data, test_data = train_test_split(df, test_size=testscale)
	return train_data,test_data

# 数据的 [用户-产品] 矩阵
def getUserProductMatrix(data,row,col):
	data_matrix = np.zeros((row, col))
	for line in data.itertuples():
	    data_matrix[line[1]-1, line[2]-1] = line[3]
	return data_matrix

# 计算余弦相似性
def calCostDistance(data_matrix):
	return pairwise_distances(data_matrix,metric='cosine')

# 基于内容的协同过滤：两部分，一是物品，一是用户
def predict(ratings, similarity, type='user'):
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1)
            #You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
# ------------------------------------#

df = readDate().iloc[:,0:3]
num_users = getNumberOfUsers(df)
num_videos  = getNumberOfVideos(df)

print ('Number of users = ' + str(getNumberOfUsers(df)) + ' | Number of movies = ' + str(getNumberOfVideos(df)))
train_data, test_data = getTrainAndTestDate(df,0.25)

# 梯度下降法
def costFunction(params,Y,num_users,num_videos,num_features,lambd):
	X = params[0:num_videos*num_features].reshape((num_videos,num_features))
	Theta = params[num_videos*num_features:].reshape((num_users,num_features))
	isRating = np.where(Y > 0, 1, 0)
	# print(X.shape)
	# print(Theta.shape)
	J = 1/2 * np.sum(np.multiply(isRating,(np.dot(Theta,X.T)-Y)**2)) + lambd/2* np.sum(X**2) + lambd/2 * np.sum(Theta**2)
	return J
# 偏导数
def gradient(params, Y,num_users,num_videos,num_features,lambd):

	X = params[0:num_videos*num_features].reshape((num_videos,num_features))
	Theta = params[num_videos*num_features:].reshape((num_users,num_features))
	isRating = np.where(Y > 0, 1, 0)

	X_grad = np.zeros(X.shape)
	Theta_grad = np.zeros(Theta.shape)

	X_grad = np.dot( np.multiply(isRating,(np.dot(Theta, X.T) - Y)).T, Theta) + np.dot(lambd,X)
	Theta_grad = np.dot(np.multiply(isRating,(np.dot(Theta, X.T) - Y)),X) + np.dot(lambd,Theta)

	return np.row_stack((X_grad.reshape((num_videos*num_features,1)),Theta_grad.reshape((num_users*num_features,1)))).flatten()

# 标准化
def normalizeRating(Y):
	m,n = Y.shape
	Ymean = np.zeros((1,n))
	Ynorm = np.zeros(Y.shape)
	for i in range(0,n):
		idx = np.where(Y[:,i]>0)
		Ymean[0,i] = np.mean(Y[idx,i])
		Ynorm[idx,i] = (Y[idx,i] - Ymean[0,i])/5.0
	return Ynorm

#隐藏因子数量
num_features = 10
#正则化系数
lambd = 1

# 两个待求矩阵
X = np.random.randn(num_videos,num_features) / np.sqrt(num_features)
Theta = np.random.randn(num_users,num_features) / np.sqrt(num_features)

print(X.shape)
print(Theta.shape)

# 向量化
# X_vector = X.reshape((num_users*num_features,1))
# Theta_vector = Theta.reshape((num_users*num_features,1))

# 向量化、合并，用于梯度运算
initial_Theta = np.row_stack((X.reshape((num_videos*num_features,1)),Theta.reshape((num_users*num_features,1)))).flatten()

print(initial_Theta.shape)

# print(np.dot(initial_Theta,initial_Theta))

# 评分矩阵,Y = Theta*X.T
Y = getUserProductMatrix(train_data,num_users,num_videos)

cost = costFunction(initial_Theta,normalizeRating(Y),num_users,num_videos,num_features,0)
# grad = gradient(initial_Theta,normalizeRating(Y),num_users,num_videos,num_features,lambd)

print("cost = ",cost)
Result = scop.minimize(fun=costFunction, x0=initial_Theta, args=(normalizeRating(Y),num_users,num_videos,num_features,lambd), method='CG', jac=gradient,\
					   options={'disp': True,'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 300})
# optimalTheta = scop.fmin_cg(f=costFunction, x0=initial_Theta, fprime=gradient,args=(Y,num_users,num_videos,num_features,0),disp=True, maxiter=500)

optimalTheta = Result.x
print('Cost at theta found by fminunc:', Result.fun)
print('theta: \n', optimalTheta)

cost = costFunction(optimalTheta,normalizeRating(Y),num_users,num_videos,num_features,lambd)
print("cost:",cost)
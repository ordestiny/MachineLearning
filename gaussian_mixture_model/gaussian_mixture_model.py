# 高斯混合模型

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from sklearn import mixture

n_sample = 300

np.random.seed(0)

shifted_gaussian = np.random.randn(n_sample,2) + np.array([20,20])

C = np.array([[0.,-0.7],[3.5,.7]])
stretched_gaussian = np.dot(np.random.randn(n_sample,2),C)

X_train = np.vstack([shifted_gaussian,stretched_gaussian])

clf = mixture.GaussianMixture(n_components=2,covariance_type="full")
clf.fit(X_train)

x = np.linspace(-20.,40.)
y = np.linspace(-20.,40.)

X,Y = np.meshgrid(x,y)
XX = np.array([X.ravel(),Y.ravel()]).T
Z =  -clf.score_samples(XX)
Z = Z.reshape(X.shape)

# 绘等高线
CS = plt.contour(X,Y,Z,norm = LogNorm(vmin=1.0,vmax=1000.0),levels = np.logspace(0,3,10))
# 显示色柱
CB = plt.colorbar(CS,shrink = 0.8,extend= "both" )
plt.scatter(X_train[:,0],X_train[:,1],.8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
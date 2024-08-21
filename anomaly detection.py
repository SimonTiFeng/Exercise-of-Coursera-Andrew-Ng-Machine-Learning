import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

Data = loadmat('ex8data1.mat')
X = Data['X']
Xval = Data['Xval']
yval = Data['yval']

Xtrain = X[0:300, :]
ytrain = np.ones(300)

def GaussianDistribution(X, mu, sigma):
    tem = np.exp(-(X - mu)**2 / (2 * sigma**2))
    return np.prod(tem / (np.sqrt(2 * np.pi) * sigma),axis=1)

def muCalc(X):
    return np.mean(X, axis=0)

def sigmaCalc(X):
    mu = muCalc(X)
    return np.sqrt(np.sum((X - mu)**2, axis=0) / (X.shape[0]))

def plotData(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

proba = GaussianDistribution(Xtrain, muCalc(Xtrain), sigmaCalc(Xtrain))

x_feature = Xtrain[:, 0]  
y_feature = Xtrain[:, 1]  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_feature, y_feature, proba, c=proba, cmap='viridis', marker='o')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Predicted Probability')
ax.set_title('3D Scatter Plot of Xtrain and Predicted Probability')

plt.show()

print(GaussianDistribution(X[301:307,:], muCalc(Xtrain), sigmaCalc(Xtrain)))

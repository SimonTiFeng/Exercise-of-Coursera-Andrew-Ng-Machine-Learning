import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import scipy.optimize as opt

# Load movie ratings data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
nmovies, nusers = Y.shape
t = 3

def gradient(XandTheta, Y, R):
    Lambda = 10
    nmovies, nusers = Y.shape
    t = 3

    X = XandTheta[:nmovies * t]
    Theta = XandTheta[nmovies * t:]
    X = X.reshape(nmovies, t)
    Theta = Theta.reshape(nusers, t)

    PredY = X.dot(Theta.T)
    Error = R * (PredY - Y)
    gradX = Error.dot(Theta) + Lambda * X
    gradTheta = Error.T.dot(X) + Lambda * Theta

    return np.concatenate((gradX.reshape(-1), gradTheta.reshape(-1)))

def cost(XandTheta, Y, R):
    Lambda = 10
    X = XandTheta[:nmovies*t]
    Theta = XandTheta[nmovies*t:]
    X = X.reshape(nmovies, t)
    Theta = Theta.reshape(nusers, t)

    PredY = X.dot(Theta.T)
    Error = R * (PredY - Y)
    J = (1.0 / 2) * np.sum(Error ** 2) + (Lambda / 2) * (np.sum(Theta ** 2) + np.sum(X ** 2))
    return J

# Initialize X and Theta
X = np.random.randn(nmovies, t)
Theta = np.random.randn(nusers, t)
XandTheta = np.concatenate((X.reshape(-1), Theta.reshape(-1)))

# Set hyperparameters
Lambda = 10


XandTheta = np.concatenate((X.reshape(-1), Theta.reshape(-1)))
XandTheta = opt.minimize(fun = cost, x0=XandTheta, args=(Y, R),method = 'TNC', jac=gradient, options={'maxiter': 1000, 'disp': True})
X_opt = XandTheta.x[:nmovies * t].reshape(nmovies, t)
Theta_opt = XandTheta.x[nmovies * t:].reshape(nusers, t)

# Predicted ratings
Y_pred = X_opt.dot(Theta_opt.T)

# Visualize the predicted ratings and original ratings
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(Y, ax=ax[0], cmap="YlGnBu")
ax[0].set_title('Original Ratings')

sns.heatmap(Y_pred, ax=ax[1], cmap="YlGnBu")
ax[1].set_title('Predicted Ratings')

plt.show()

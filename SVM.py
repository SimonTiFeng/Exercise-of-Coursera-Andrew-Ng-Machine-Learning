import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
import scipy.optimize as opt

data = loadmat('ex6data1.mat')
X = data['X']
y = data['y']
X = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.zeros(X.shape[0])

def CostFunction(theta, kernel_matrix, y):
    print(kernel_matrix.shape)
    kernel_matrix = np.matrix(kernel_matrix)
    y = np.matrix(y)
    theta = np.matrix(theta).reshape(-1, 1) 
    first = np.multiply(y,np.maximum(0 , np.dot(kernel_matrix,theta) + 1))
    second = np.multiply(1-y,np.maximum(0, -np.dot(kernel_matrix,theta) + 1))
    regularization = (theta[1:].T * theta[1:]).sum() * 8.61019102
    J = (first - second).sum() / (2 * kernel_matrix.shape[0]) + regularization
    return J

def kernel(X,sigma=1.0):
    n_samples = X.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i, 1:] - X[j, 1:]  
            distance_squared = np.dot(diff, diff)
            kernel_matrix[i, j] = np.exp(-distance_squared / (2 * sigma**2))
    return kernel_matrix

def judge(kernel_matrix,theta):
    kernel_matrix = np.matrix(kernel_matrix)
    theta = np.matrix(theta).reshape(-1, 1) 
    result = np.sign(np.dot(kernel_matrix,theta))
    return result.flatten()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def gradient(theta, kernel_matrix, y):
    theta = np.matrix(theta).reshape(-1, 1) 
    kernel_matrix = np.matrix(kernel_matrix)
    y = np.matrix(y)
    
    parameters = theta.size
    grad = np.zeros(parameters)
    
    error = sigmoid(kernel_matrix * theta) - y
    
    for i in range(parameters):
        term = np.multiply(error, kernel_matrix[:,i])
        grad[i] = np.sum(term) / len(kernel_matrix)
    
    return grad

kernel_matrix = kernel(X, sigma=1.0)
print(kernel_matrix.shape)
result = opt.minimize(CostFunction, x0=theta, jac=gradient, args=(kernel_matrix,y))
predictions = judge(kernel_matrix, result.x)
predictions = (predictions > 0).astype(int)
accuracy = np.mean(predictions == y.flatten()) * 100
print(f'模型的正确率为: {accuracy:.2f}%')




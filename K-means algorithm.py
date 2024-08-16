import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
import random

# The definition input an array and output a number
def dist_calc(X, centroids):
    distances = X - centroids
    distances = np.sum(distances**2) 
    return np.sqrt(distances) 

# The definition output an array of index
def find_closet_centroid(X, centroids):
    index = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        min_distance = 1000000

        for j in range(centroids.shape[0]):
            dist = dist_calc(X[i], centroids[j])
            if dist < min_distance:
                min_distance = dist
                index[i] = j

    return index

# The data set of K-means algorithm
data = loadmat('ex7data2.mat')  
X = data['X']
K = 3

def run_kmeans(X, K, num_iterations=10):
    best_centroids = None
    best_index = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        # Initialize centroids randomly
        centroids = X[random.sample(range(X.shape[0]), K)]

        for _ in range(10):  # 每次迭代运行10次
            index = find_closet_centroid(X, centroids)

            # 计算每个簇的数据点数量
            counts = np.bincount(index, minlength=K)

            # 更新质心
            new_centroids = np.zeros((K, 2))
            for j in range(X.shape[0]):
                new_centroids[index[j]] += X[j]

            for k in range(K):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]

            centroids = new_centroids

        # 计算当前聚类的代价（误差平方和）
        cost = 0
        for j in range(X.shape[0]):
            cost += dist_calc(X[j], centroids[index[j]])**2

        if cost < best_cost:
            best_cost = cost
            best_centroids = centroids
            best_index = index

    return best_centroids, best_index

# 运行 K-means 算法多次并选择最优结果
best_centroids, best_index = run_kmeans(X, K, num_iterations=10)

# 绘制结果
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b']
for i in range(K):
    plt.scatter(X[best_index == i, 0], X[best_index == i, 1], s=50, c=colors[i], label=f'Cluster {i}')
plt.scatter(best_centroids[:, 0], best_centroids[:, 1], s=200, c='k', marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

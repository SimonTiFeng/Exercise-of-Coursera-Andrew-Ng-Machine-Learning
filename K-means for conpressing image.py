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
    index = np.zeros(X.shape[0],dtype=int)
    for i in range(X.shape[0]):
        min_distance = 1000000

        for j in range(centroids.shape[0]):
            dist = dist_calc(X[i], centroids[j])
            if dist < min_distance:
                min_distance = dist
                index[i] = j

    return index


image_data = loadmat('bird_small.mat')
X = image_data['A']
X = X/255
X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
X = X - np.mean(X, axis=0)
K= 16

# Initialize centroids randomly
centroids = X[random.sample(range(X.shape[0]), K)]

# K-means algorithm
index = find_closet_centroid(X, centroids)

index_Class = [[] for _ in range(K)]
for i, idx in enumerate(index):
    index_Class[int(idx)].append(i)


def Cal_index(index_class,K):
    result = np.bincount(index_class, minlength=K)
    return result

# K-means update centroids
centroids = np.zeros((K, X.shape[1]))
counts = np.bincount(index, minlength=K)
for j in range(X.shape[0]):
    centroids[index[j]] += X[j]
for k in range(K):
    if counts[k] > 0:
        centroids[k] /= counts[k]

# K-means visualization
def visualize_clusters(X, centroids, index, K):
    plt.figure(figsize=(10, 6))
    colors = sb.color_palette("hls", K)
    for k in range(K):
        cluster_points = X[index == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], label=f'Cluster {k+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def compress_image(X, centroids, index, original_shape):
    compressed_image = centroids[index]
    compressed_image = np.reshape(compressed_image, original_shape)
    return compressed_image

# acquire shape of original image
original_shape = image_data['A'].shape

# compress image
compressed_image = compress_image(X, centroids, index, original_shape)

# image visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_data['A'] / 255)
plt.title('Original Image')
plt.axis('off')
plt.figure(figsize=(6, 6))
plt.imshow(compressed_image)
plt.title('Compressed Image')
plt.axis('off')
plt.show()
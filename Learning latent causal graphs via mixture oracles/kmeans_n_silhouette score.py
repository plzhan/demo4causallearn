# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     kmeans_n_silhouette score
# @Time:         14:48  2023/11/30
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 生成合成数据
X, y = make_blobs(n_samples=1000, centers=5, random_state=42)

# 初始化聚类簇数范围
cluster_range = range(2, 10)
scores = []
fig, axs = plt.subplots(3,3, figsize=(16, 16))
# 迭代不同的聚类簇数
for i, n_clusters in enumerate(cluster_range):
    # 执行K均值聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X)

    # 计算轮廓系数
    score = silhouette_score(X, labels)
    scores.append(score)

    # 绘制聚类结果
    axs[i // 3, i % 3].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axs[i // 3, i % 3].set_title(f'Clustering Result for {n_clusters} Clusters')

fig.suptitle('Hierarchical Clustering Results(K-Means)', fontsize=32)
# 绘制折线图
i+=1
axs[i // 3, i % 3].plot(cluster_range, scores, marker='o')
axs[i // 3, i % 3].set_xlabel('Number of Clusters')
axs[i // 3, i % 3].set_ylabel('Silhouette Score')
axs[i // 3, i % 3].set_title('Silhouette Score for Different Cluster Numbers')
plt.savefig('Hierarchical Clustering Results(K-Means).png', dpi=200)
plt.show()

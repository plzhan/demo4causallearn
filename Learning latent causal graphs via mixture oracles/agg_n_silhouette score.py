# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     agg_n_silhouette score
# @Time:         14:53  2023/11/30
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 生成合成数据
X, y = make_blobs(n_samples=1000, centers=5, random_state=42)

# 设置聚类数量的范围
n_clusters_range = range(2, 10)

# 存储每个聚类数量的轮廓系数分数
silhouette_scores = []

# 创建子图布局
fig, axs = plt.subplots(3,3, figsize=(16, 16))

# 遍历聚类数量范围
for i, n_clusters in enumerate(n_clusters_range):
    # 执行层次聚类
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_clustering.fit_predict(X)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    print(f'For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}')

    # 绘制数据的散点图
    axs[i // 3, i % 3].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axs[i // 3, i % 3].set_title(f'Clustering Result for {n_clusters} Clusters')

fig.suptitle('Hierarchical Clustering Results(AgglomerativeClustering)', fontsize=32)
# 绘制折线图显示轮廓系数分数的变化
i+=1
axs[i // 3, i % 3].plot(n_clusters_range, silhouette_scores, marker='o')
axs[i // 3, i % 3].set_xlabel('Number of Clusters')
axs[i // 3, i % 3].set_ylabel('Silhouette Score')
axs[i // 3, i % 3].set_title('Silhouette Score for Different Cluster Numbers')
plt.savefig('Hierarchical Clustering Results(AgglomerativeClustering).png', dpi=200)
plt.show()

# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     SELF
# @Time:         20:02  2023/11/14
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity


class FunctionSet(object):
    def __init__(self):
        self.F = dict()

    def __getitem__(self, item):
        return self.F[f'{item}']

    def __setitem__(self, key, value: XGBRegressor):
        self.F[f'{key}'] = value

    def __len__(self):
        return len(self.F.keys())

    def get_di(self, key):
        return len(self.F[f'{key}'].get_score(importance_type='weight').keys())

class ParentCache(FunctionSet):
    def __init__(self):
        super(ParentCache, self).__init__()

    def __setitem__(self, key, value: list):
        self.F[f'{key}'] = set(value)

    def __getitem__(self, item):
        try:
            return list(self.F[f'{item}'])

        except Exception:
            return None

    def get_di(self, key):
        pass


class Graph(object):
    def __init__(self, n):
        self.graph = np.zeros((n, n))

    def set_edge(self, x, y):
        assert x!=y
        self.graph[x, y] = 1
        self.graph[y, x] = -1

    def remove_edge(self, x, y):
        assert x!=y
        self.graph[x, y] = 0
        self.graph[y, x] = 0

    def reverse_edge(self, x, y):
        assert x!=y
        self.graph[x, y], self.graph[y, x] = self.graph[y, x], self.graph[x, y]

    def get_x_parent(self, x):
        index = np.where(self.graph[x, :]==1)[0]
        index = index.tolist()
        return index if index != [] else None

class SELF(object):
    def __init__(self, dataset, is_split=True):
        self.dataset = dataset
        self.m = len(dataset)
        self.functionSet = FunctionSet()
        self.graph = Graph(self.m)
        self.Likelihood = [0 for i in range(self.m)]
        self.is_split = is_split
        self.parent_cache = FunctionSet()

    def compute_sum_log_i(self, i):
        bicterm = self.compute_bic_i(i)
        parent = self.graph.get_x_parent(i)
        # 如果是没有父节点
        if not parent:  # 初始化
            # 那就当前节点的数值进行计算L
            Ei = self.dataset[:, i][:, None]
        # 有父亲节点就要召唤回归器了
        else:
            if parent != self.parent_cache[i]:
                self.parent_cache[i] = parent
                self.functionSet[i] = self.get_xgboost_model()

            oi = self.dataset[:, i][:, None]
            parenti = self.dataset[:, parent]
            if self.is_split:
                parenti_train, parenti_test, oi_train, oi_test = train_test_split(parenti, oi, test_size=0.2, random_state=42)
            else:
                parenti_train = parenti
                oi_train = oi
            self.functionSet[i].fit(parenti_train, oi_train)
            Fi = self.functionSet[i].predict(parenti)
            Ei = oi - Fi

        Pri = self.get_kernel_density_model(Ei)
        self.Likelihood[i] = np.sum(Pri) - bicterm

    def compute_bic_i(self, i):
        di = self.functionSet.get_di(i)
        return di*np.log(self.m) / self.m / 2

    def init_self(self):

        pass

    def get_xgboost_model(self):
        model = XGBRegressor()
        return model

    def silverman_bandwidth(self, xi):
        n = xi.shape[0]
        std = xi.std()
        bw = 1.06 * std * n ** (-0.2)
        return bw

    def get_kernel_density_model(self, Ei):
        bw = self.silverman_bandwidth(Ei)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(Ei)
        return kde.score_samples(Ei)

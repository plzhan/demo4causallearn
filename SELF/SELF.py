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
        try:
            return len(self.F[f'{key}'].get_score(importance_type='weight').keys())
        except Exception:
            return 0


class ParentCache(FunctionSet):
    """
    用于存放每个节点出现过的父亲节点对应的log值(所有样本求和后的)
    当作一个高速缓存，不用重新计算浪费时间。

    相当于Δ(G, G`)中，只需要update父亲节点更变的节点i,其他都只是O(1)的访问时间
    """
    def __init__(self):
        super(ParentCache, self).__init__()

    def __setitem__(self, key, value: set):
        assert type(value) is set, "TypeError: type of value should be set"
        self.F.setdefault(f'{key}', dict())
        self.F[f'{key}'][f'{value}'] = None

    def __getitem__(self, item):
        try:
            return self.F[f'{item}']

        except Exception:
            return None

    def get_i_pi_sum_log(self, i, pi: set):
        return self.F[f'{i}'][f'{pi}']

    def store_pi_sum_log(self, i, pi: set, sum_log):
        assert type(pi) is set, "TypeError: type of value should be set"
        self.F[f'{i}'][f'{pi}'] = sum_log

    def check_pi_in_i(self, i, pi: set):
        assert type(pi) is set, "TypeError: type of value should be set"
        return f'{pi}' in self.F[f'{i}']

    def get_di(self, key):
        pass


class Graph(object):
    def __init__(self, n):
        self.graph = np.zeros((n, n))

    def set_edge(self, x, y):
        assert x != y
        self.graph[x, y] = 1
        self.graph[y, x] = -1

    def remove_edge(self, x, y):
        assert x != y
        self.graph[x, y] = 0
        self.graph[y, x] = 0

    def reverse_edge(self, x, y):
        assert x != y
        self.graph[x, y], self.graph[y, x] = self.graph[y, x], self.graph[x, y]

    def get_x_parent(self, x):
        index = np.where(self.graph[x, :] == 1)[0]
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
        self.parent_cache = ParentCache()

    def compute_sum_log_i(self, graph, i):
        parent = graph.get_x_parent(i)
        bicterm = 0
        function_i = None
        # 如果是没有父节点
        if self.parent_cache.check_pi_in_i(i, set(parent)):
            return self.parent_cache.get_i_pi_sum_log(i, set(parent)), function_i

        if not parent:  # 初始化
            # 那就当前节点的数值进行计算L
            parent = [i]
            Ei = self.dataset[:, [parent]]
        # 有父亲节点就要召唤回归器了
        else:
            function_i = self.get_xgboost_model()

            oi = self.dataset[:, i][:, None]
            parenti = self.dataset[:, parent]
            if self.is_split:
                parenti_train, parenti_test, oi_train, oi_test = train_test_split(parenti, oi, test_size=0.2,
                                                                                  random_state=42)
            else:
                parenti_train = parenti
                oi_train = oi
            function_i.fit(parenti_train, oi_train)
            Fi = function_i.predict(parenti)
            Ei = oi - Fi
            bicterm = self.compute_bic(function_i)
        Pri = self.get_kernel_density_model(Ei)
        likelihood = np.sum(Pri) / self.m - bicterm
        self.parent_cache[i] = set(parent)
        self.parent_cache.store_pi_sum_log(i, set(parent), likelihood)

        return likelihood, function_i

    def compute_graph_likelihood(self, graph):
        n_nodes = self.m
        tt_likelihood = 0
        for i in range(n_nodes):
            tt_likelihood += self.compute_sum_log_i(graph, i)
        return tt_likelihood

    def compute_bic(self, function):
        di = len(function.get_score(importance_type='weight').keys())
        return di * np.log(self.m) / self.m / 2

    def search_neighbor_graph(self):


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

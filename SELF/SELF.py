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


class FunctionSet(object):
    def __init__(self):
        self.F = dict()

    def __getitem__(self, item):
        return self.F[f'{item}']

    def __setitem__(self, key, value: XGBRegressor):
        assert value
        self.F[f'{key}'] = value

    def __len__(self):
        return len(self.F.keys())

    def get_di(self, key):
        return len(self.F[f'{key}'].get_score(importance_type='weight').keys())

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
    def __init__(self, dataset):
        self.dataset = dataset
        self.m = len(dataset)
        self.functionSet = FunctionSet()
        self.graph = Graph(self.m)

    def compute_sum_log_i(self, i):
        parent = self.graph.get_x_parent(i)
        if not parent:


    def compute_bic_i(self, i):
        di = self.functionSet.get_di(i)
        return di*np.log(self.m) / self.m / 2

    def init_self(self):

        pass

    def get_xgboost_model(self):
        model = XGBRegressor()
        return model



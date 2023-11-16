# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     SELF
# @Time:         20:02  2023/11/14
from copy import deepcopy
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import graphviz

class FunctionSet(object):
    """
    version 1.x 版本 没有返回Function(每个结点父类的模型)
    有空时后续会加上
    """
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
        assert isinstance(pi, set), "TypeError: type of value should be set"
        self.F[f'{i}'][f'{pi}'] = sum_log

    def check_pi_in_i(self, i, pi: set):
        assert isinstance(pi, set), "TypeError: type of value should be set"
        try:
            return f'{pi}' in self.F[f'{i}']
        except:
            return False

    def get_di(self, key):
        pass


class Graph(object):
    def __init__(self, n: int or np.array):
        if isinstance(n, int):
            self.graph = np.zeros((n, n))
        else:
            self.graph = n

    def check_edge(self, x, y):
        return self.graph[x, y]

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
        index = np.where(self.graph[x, :] == -1)[0]
        index = index.tolist()
        return index if index != [] else []


class SELF(object):
    def __init__(self, dataset, is_split=True):
        self.dataset = dataset
        self.n, self.m = dataset.shape[0], dataset.shape[1]
        self.functionSet = FunctionSet()
        self.graph = Graph(self.m)  # max-likelihood
        self.likelihood = 0
        self.likelihood_star = 0
        self.is_split = is_split
        self.parent_cache = ParentCache()
        self.neighbor_graphs = []

    def compute_sum_log_i(self, graph: Graph, i):
        parent = graph.get_x_parent(i)
        n_parent = len(parent)
        oi = self.dataset[:, i][:, None]
        di = 0  # 无父节点的时候没有模型就没有模型复杂度

        if self.parent_cache.check_pi_in_i(i, set(parent)):
            return self.parent_cache.get_i_pi_sum_log(i, set(parent))

        if n_parent == 0:  # 初始化
            # 那就当前节点的数值进行计算L
            parent = [i]
            Ei = self.dataset[:, parent]

        # 有父亲节点就要召唤回归器了
        else:
            function_i = self.get_xgboost_model()
            parenti = self.dataset[:, parent]
            if self.is_split:
                parenti_train, parenti_test, oi_train, oi_test = train_test_split(parenti, oi, test_size=0.2,
                                                                                  random_state=42)
            else:
                parenti_train = parenti
                oi_train = oi

            function_i.fit(parenti_train, oi_train)
            Fi = function_i.predict(parenti)[:, None]
            Ei = oi - Fi
            di = self.compute_model_complexity(function_i, n_parent)

        Pri = self.get_kernel_density_model(Ei)
        bicterm = di * np.log(self.n) / self.n / 2

        A, B = np.sum(Pri) / self.n, bicterm
        # print("\nPri:", A, "bic:", B)
        likelihood = A - B
        self.parent_cache[i] = set(parent)
        self.parent_cache.store_pi_sum_log(i, set(parent), likelihood)

        return likelihood

    def compute_model_complexity(self, function: XGBRegressor, n_parent):
        num_estimator = function.n_estimators
        num_leaves = sum(1 for tree in function.get_booster().get_dump() for i in tree.split(',') if 'leaf' in i)
        di = num_leaves * num_estimator + n_parent
        return di

    def compute_graph_likelihood(self, graph: Graph):
        n_nodes = self.m
        tt_likelihood = 0
        # functions = []
        for i in range(n_nodes):
            # print(i)
            tt_likelihood += self.compute_sum_log_i(graph, i)
            # print("\n", tt_likelihood)
            # functions.append(function_i)
        return tt_likelihood

    def search_neighbor_graphs(self, graph: np.array):
        neighbor_graphs = []
        for i in range(self.m-1):
            for j in range(i+1, self.m):
                if i != j:
                    graph_copy = Graph(self.m)
                    graph_copy.graph = graph
                    if graph_copy.check_edge(i, j) == 0:
                        graph_copy.set_edge(i, j)
                        neighbor_graphs.append(graph_copy.graph.copy())
                        graph_copy.remove_edge(i, j)
                        continue

                    if graph_copy.check_edge(i, j) == 1:
                        graph_copy.reverse_edge(i, j)
                        neighbor_graphs.append(graph_copy.graph.copy())
                        graph_copy.reverse_edge(i, j)
                        continue

                    if graph_copy.check_edge(i, j) == -1:
                        graph_copy.remove_edge(i, j)
                        neighbor_graphs.append(graph_copy.graph.copy())
                        graph_copy.set_edge(j, i)
                        continue
        return neighbor_graphs

    def hill_climbing_based_causal_structure_search(self):

        pbar = tqdm(total=None)
        pbar.set_description(desc=">> Compute Old Graph's Likelihood...")
        self.likelihood = self.compute_graph_likelihood(self.graph)
        pbar.set_description(
            desc="Old likelihood: L = " + str(round(self.likelihood, 3)) + "\n>> Start Hill Climbing...")

        Stage = 0
        while True:
            Stage += 1
            pbar.set_description(desc=f"\n==============Stage: {Stage}==============")
            self.neighbor_graphs = self.search_neighbor_graphs(self.graph.graph)
            pbar.set_description(desc="\n>> Found A Group of Neighbor Graphs!")
            likelihood_max_iter = [0, self.likelihood]
            for i, graph_i in enumerate(self.neighbor_graphs):
                pbar.set_description(desc=f"\n>> Computing Neighbor Graph {i}")
                likelihood = self.compute_graph_likelihood(Graph(graph_i))
                likelihood_max_iter = [i, likelihood] if likelihood > likelihood_max_iter[1] else likelihood_max_iter
            if self.likelihood < likelihood_max_iter[1]:
                pbar.set_description(desc="\n>> Larger likelihood: L = " + str(round(likelihood_max_iter[1], 3)))
                print("\n>>>>>> Found larger likelihood: L = " + str(round(likelihood_max_iter[1], 3)), " <<<<<<")
                self.likelihood = likelihood_max_iter[1]
                self.graph.graph = self.neighbor_graphs[likelihood_max_iter[0]]
            else:
                break

    def get_xgboost_model(self):
        model = XGBRegressor(n_estimators=12)
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

def draw_graph(adj_matrix):
    dot = graphviz.Digraph(comment='The Round Table')

    n = len(adj_matrix)  # number of nodes
    # Create nodes
    for i in range(n):
        dot.node(str(i))

    # Create edges
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:  # there is a directed edge from i to j
                dot.edge(str(i), str(j))

    # Render the graph
    dot.format = 'png'
    dot.render('output_graph', view=True)

if __name__ == '__main__':
    # x->y->z linear data
    np.random.seed(0)
    x = np.random.normal(size=4000)
    y = 3 * x + np.random.uniform(-1, 1, size=4000) * 0.1
    z = 3 * y + np.random.uniform(-1, 1, size=4000) * 0.1
    z1 = 4 * x + np.random.uniform(-1, 1, size=4000) * 0.1

    data = np.column_stack((x, y, z, z1))

    SELF_model = SELF(dataset=data, is_split=True)
    SELF_model.hill_climbing_based_causal_structure_search()
    print("\nRESULT:")
    print(SELF_model.graph.graph)
    draw_graph(SELF_model.graph.graph)

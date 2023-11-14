# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     PC_Model_Repetition
# @Time:         19:08  2023/10/20
import copy
from math import sqrt
import numpy as np
from scipy.stats import norm
from causallearn.graph.GraphClass import CausalGraph, Edge, Endpoint
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery
from causallearn.utils.PCUtils.Helper import append_value
from itertools import combinations
from causallearn.utils.cit import CIT, CIT_Base, FisherZ


class PcModel(object):

    def __init__(self, data=None, alpha=0.05, background_knowledge=None):
        self.data = data
        self.cg = None
        self.alpha = alpha
        self.data_shape = None if data is None else data.shape
        self.p = np.corrcoef(self.data.T)  # n_samples, n_features
        self.indep_test = CIT(data, "fisherz")

    def build_completed_graph(self, data=None):
        if data is not None:
            self.data = data
            self.data_shape = data.shape
        n_nodes = self.data.shape[1]
        self.cg = CausalGraph(n_nodes, None)
        self.cg.draw_pydot_graph()

    def fisher_z_test(self, x_and_y_and_condition=None):
        x, y, z = [i for i in x_and_y_and_condition]
        x_and_y_and_condition = [x] + [y] + list(z)

        sub_p = self.p[np.ix_(x_and_y_and_condition, x_and_y_and_condition)]
        inv = np.linalg.inv(sub_p)
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r)
        Z = 0.5 * np.log((1 + r) / (1 - r))
        x = np.abs(Z) * np.sqrt(self.data_shape[0] - len(z) - 3)
        p = 2 * (1 - norm.cdf(np.abs(x)))
        return p

    def get_adj_i_except_j(self, i, l):
        adj_i = self.cg.neighbors(i)
        return adj_i if len(adj_i) >= l else None

    def build_skeleton(self):
        self.cg.set_ind_test(self.indep_test)
        c = FisherZ(self.data, )
        l = 0
        while self.cg.max_degree() - 1 >= l:
            for i in range(self.data_shape[1]):
                adj_i = self.get_adj_i_except_j(i, l)
                if adj_i is None:
                    continue
                for j in adj_i:
                    adj_i_j = np.delete(adj_i, np.where(adj_i == j))

                    for k in combinations(adj_i_j, l):
                        # print(k)
                        # i, j = (i, j) if i < j else (j, i)
                        if i not in k and j not in k:
                            # i, j = (i, j) if i < j else (j, i)
                            x_and_y_and_condition = [i] + [j] + list(k)
                        else:
                            continue
                        # print(x_and_y_and_condition)
                        # 相关性p越小，双边概率越大，双边概率 > alpha 表示两节点独立，拒绝的是两节点关联的假设，
                        # p = c(i, j, k)  # x and y are int
                        x_and_y_and_condition = (i, j, k)
                        # print(x_and_y_and_condition)
                        if self.fisher_z_test(x_and_y_and_condition) > self.alpha:
                        # if p > self.alpha:

                            edge1 = self.cg.G.get_edge(self.cg.G.nodes[i], self.cg.G.nodes[j])
                            edge2 = self.cg.G.get_edge(self.cg.G.nodes[j], self.cg.G.nodes[i])

                            if edge1 is not None:
                                self.cg.G.remove_edge(edge1)
                                append_value(self.cg.sepset, i, j, k)

                            if edge2 is not None:
                                self.cg.G.remove_edge(edge2)
                                append_value(self.cg.sepset, j, i, k)

                            break
            l += 1

        self.cg.draw_pydot_graph()

    def build_directed_acyclic_graph(self):
        # 首先是找到V结构，不重复的
        copy_cg = copy.deepcopy(self.cg)
        UT = [(i, j, k) for (i, j, k) in copy_cg.find_unshielded_triples() if i < k]

        # step C
        for x, y, z in UT:
            # check collider
            if all(y not in S for S in copy_cg.sepset[x, z]):
                # y is not in the separation set，then collider will be defined.
                # firstly, we have to check if the edge is oriented
                edge1 = copy_cg.G.get_edge(copy_cg.G.nodes[x], copy_cg.G.nodes[y])
                edge2 = copy_cg.G.get_edge(copy_cg.G.nodes[y], copy_cg.G.nodes[x])
                if edge1 is not None:
                    copy_cg.G.remove_edge(edge1)
                if edge2 is not None:
                    copy_cg.G.remove_edge(edge2)

                copy_cg.G.add_edge(Edge(copy_cg.G.nodes[x], copy_cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge3 = copy_cg.G.get_edge(copy_cg.G.nodes[z], copy_cg.G.nodes[y])
                edge4 = copy_cg.G.get_edge(copy_cg.G.nodes[y], copy_cg.G.nodes[z])
                if edge3 is not None:
                    copy_cg.G.remove_edge(edge3)
                if edge4 is not None:
                    copy_cg.G.remove_edge(edge4)

                copy_cg.G.add_edge(Edge(copy_cg.G.nodes[z], copy_cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

        # return Meek.meek(copy_cg, background_knowledge=None)
        # Step D)
        # Situation: i -> j o-o k
        UT = copy_cg.find_unshielded_triples()
        Tri = copy_cg.find_triangles()
        Kite = copy_cg.find_kites()
        loop = True
        while loop:
            loop = False
            for i, j, k in UT:
                if copy_cg.is_fully_directed(i, j) and copy_cg.is_undirected(j, k):
                    edge1 = copy_cg.G.get_edge(copy_cg.G.nodes[j], copy_cg.G.nodes[k])
                    if edge1 is not None:
                        if copy_cg.G.is_ancestor_of(copy_cg.G.nodes[k], copy_cg.G.nodes[j]):
                            continue
                        else:
                            copy_cg.G.remove_edge(edge1)
                    else:
                        continue

                    copy_cg.G.add_edge(Edge(copy_cg.G.nodes[j], copy_cg.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    loop = True

            for i, j, k in Tri:
                if copy_cg.is_fully_directed(i, j) and copy_cg.is_fully_directed(j, k) and copy_cg.is_undirected(k, i):
                    edge1 = copy_cg.G.get_edge(copy_cg.G.nodes[i], copy_cg.G.nodes[k])
                    if edge1 is not None:
                        if copy_cg.G.is_ancestor_of(copy_cg.G.nodes[i], copy_cg.G.nodes[k]):
                            copy_cg.G.remove_edge(edge1)
                        else:
                            continue
                    else:
                        continue
                    copy_cg.G.add_edge(Edge(copy_cg.G.nodes[i], copy_cg.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    loop = True

            for i, j, k, l in Kite:
                if copy_cg.is_undirected(i, j) and copy_cg.is_undirected(i, l) and copy_cg.is_fully_directed(j, k) \
                        and copy_cg.is_fully_directed(l, k) and copy_cg.is_undirected(i, l):

                    edge1 = copy_cg.G.get_edge(copy_cg.G.nodes[i], copy_cg.G.nodes[k])
                    if edge1 is not None:
                        if copy_cg.G.is_ancestor_of(copy_cg.G.nodes[i], copy_cg.G.nodes[k]):
                            copy_cg.G.remove_edge(edge1)
                        else:
                            continue
                    else:
                        continue
                    copy_cg.G.add_edge(Edge(copy_cg.G.nodes[i], copy_cg.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    loop = True

        return copy_cg

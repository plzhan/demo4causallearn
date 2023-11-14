# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     demo1_PC
# @Time:         22:08  2023/10/10
import numpy as np
from Experiments.read_data import readdtxt
from causallearn.search.ConstraintBased.PC import pc

# read data
data_path = readdtxt(1, "auto-mpg", 'data', 0, base='../../example-causal-datasets/')
print(f"loading: {data_path}")
data = np.loadtxt(data_path, skiprows=1)  # 跳过一行

# default parameters
cg = pc(data, stable=False, uc_rule=0, uc_priority=0)
# or customized parameters
# cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)

# visualization using pydot
cg.draw_pydot_graph()

# or save the graph
from causallearn.utils.GraphUtils import GraphUtils

pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png('simple_test.png')

# visualization using networkx
# cg.to_nx_graph()
# cg.draw_nx_graph(skel=False)
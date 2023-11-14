# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     PC_Repetition
# @Time:         19:04  2023/10/20
import numpy as np
from Experiments.read_data import readdtxt
from PC_Model_Repetition import PcModel

# read data
data_path = readdtxt(1, "auto-mpg", 'data', 0, base='../../../example-causal-datasets/')
print(f"loading: {data_path}")
data = np.loadtxt(data_path, skiprows=1)  # 跳过一行

# default parameters
model = PcModel(data)
model.build_completed_graph()
model.build_skeleton()
cg = model.build_directed_acyclic_graph()


# or customized parameters
# cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)

# visualization using pydot
cg.draw_pydot_graph()

# or save the graph
from causallearn.utils.GraphUtils import GraphUtils

pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png('simple_test.png')

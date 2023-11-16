# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     anm-expriments
# @Time:         11:20  2023/11/14

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
from tqdm import tqdm
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from causallearn.search.FCMBased.ANM.ANM import ANM
def create_simulated_data(m=300, b=1, q=1, is_visualized=True):
    def f(x):
        y = x + b * x ** 3
        # y = b * np.exp(x)
        return y

    # 定义形状参数
    # 生成具有不同峰度的随机数
    x = np.random.normal(0, 1, size=(m, 1))
    n = np.random.normal(0, 1, size=(m, 1))
    absolute_value_x = np.where(x > 0, 1, -1)
    absolute_value_n = np.where(n > 0, 1, -1)
    x = np.abs(x) ** q
    n = np.abs(n) ** q
    # print((x<0).sum())
    x = x * absolute_value_x
    n = n * absolute_value_n
    # print((x<0).sum())
    data_set = [x, f(x) + n]

    x_min, x_max = np.min(x), np.max(x)
    line_x = np.arange(x_min, x_max, 0.01)
    line_y = f(line_x)
    if is_visualized:
        plt.plot(line_x, line_y, c='black')
        plt.scatter(data_set[0], data_set[1], s=3, c='black')
        plt.show()
    # return
    return data_set

def multi_repetition(times, repetition_num, b, q, is_visualized, forward, backward):
    # print("表示我正在运行，", b, q)
    while times <= repetition_num:
        anm = ANM()
        data_x, data_y = create_simulated_data(b=b, q=q, is_visualized=is_visualized)
        p_value_foward, p_value_backward = anm.cause_or_effect(data_x, data_y)
        if p_value_foward >= 0.02:
            forward += 1
        if p_value_backward >= 0.02:
            backward += 1
        times += 1
    proportion.append([q, round(forward/(repetition_num), 3), round(backward/(repetition_num), 3)])


# data_x, data_y = create_simulated_data(b=1, q=1, is_visualized=False)



# The first panel
is_visualized = False
b = 0
nums = np.arange(0.5, 2.001, 0.05)  # q
proportion = []
repetition_num = 20
with ThreadPoolExecutor(max_workers=20) as executor:
    start = time.time()
    for q in nums:
        q = round(q, 3)
        times = 1
        forward = 0
        backward = 0
        future = executor.submit(multi_repetition, times, repetition_num, b, q, is_visualized, forward, backward)
    with tqdm(total=None, desc='Progress', unit='iteration') as pbar:
        while not future.done():
            pbar.update(1)
            time.sleep(0.1)

print(f">> 执行用时: {time.time()-start}s")
plt.figure()
proportion.sort(key=lambda x: x[0])
plt.plot(nums, np.array(proportion)[:, 1:2], 'k', label='correct')
plt.plot(nums, np.array(proportion)[:, 2:], 'r', label='reverse')
plt.xlabel('q')
plt.ylabel('$p_{accept}$')
plt.title(f'b = 0')
plt.legend()
plt.savefig('b0_q05-20.jpg', dpi=200)
plt.show()

#
is_visualized = False
q = 1
nums = np.arange(-1, 1.1, 0.05)
proportion = []
repetition_num = 20
with ThreadPoolExecutor(max_workers=20) as executor:
    for b in nums:
        b = round(b, 3)
        times = 1
        forward = 0
        backward = 0
        future = executor.submit(multi_repetition, times, repetition_num, b, q, is_visualized, forward, backward)
    with tqdm(total=None, desc='Progress', unit='iteration') as pbar:
        while not future.done():
            pbar.update(1)
            time.sleep(0.1)

plt.figure()
proportion.sort(key=lambda x: x[0])
plt.plot(nums, np.array(proportion)[:, 1:2], 'k', label='correct')
plt.plot(nums, np.array(proportion)[:, 2:], 'r', label='reverse')
plt.xlabel('b')
plt.ylabel('$p_{accept}$')
plt.title(f'q = 1')
plt.legend()
# plt.show()
plt.savefig('q1_bn1-1.jpg', dpi=200)
plt.show()
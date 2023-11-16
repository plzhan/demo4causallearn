# # -*- coding:utf-8 -*-
# # Author:        zhanpl
# # Product_name:  PyCharm
# # File_name:     1
# # @Time:         16:21  2023/11/11
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
#
#
# # 定义一个函数来生成具有给定峰度的随机数
# def generate_random_numbers(n, kurtosis, c):
#     # 使用广义伽马分布生成随机数
#     """
#
#     :param n:
#     :param kurtosis: 峰度
#     :param c: 偏度
#     :return:
#     """
#     a = kurtosis
#     c = 3
#     random_numbers = stats.gengamma.rvs(a, c, size=n, random_state=42)
#
#     return random_numbers
#
#
# # 生成具有不同峰度的随机数
# random_numbers1 = generate_random_numbers(1000, 100, 70)
# random_numbers2 = generate_random_numbers(1000, 80, 70)
# random_numbers3 = generate_random_numbers(1000, 50, 70)
# random_numbers4 = generate_random_numbers(1000, 30, 70)
# random_numbers5 = generate_random_numbers(1000, 10, 70)
# random_numbers6 = generate_random_numbers(1000, 1, 70)
# random_numbers7 = generate_random_numbers(1000, 0.5, 70)
# # random_numbers = np.random.normal(mu, sigma, 1000)
#
# # 创建直方图
# # plt.hist(random_numbers, bins=30, density=True, label='Normal')
# # 创建直方图
# plt.hist(random_numbers1, bins=30, alpha=0.5, label='Kurtosis 100')
# plt.hist(random_numbers2, bins=30, alpha=0.5, label='Kurtosis 80')
# plt.hist(random_numbers3, bins=30, alpha=0.5, label='Kurtosis 50')
# plt.hist(random_numbers4, bins=30, alpha=0.5, label='Kurtosis 30')
# plt.hist(random_numbers5, bins=30, alpha=0.5, label='Kurtosis 10')
# plt.hist(random_numbers6, bins=30, alpha=0.5, label='Kurtosis 1')
# plt.hist(random_numbers7, bins=30, alpha=0.5, label='Kurtosis .1')
#
# # 添加图例和标题
# plt.legend(loc='upper right')
# plt.title('Histogram of Random Numbers with Different Kurtosis')
#
# # 显示图形
# plt.show()
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
def create_simulated_data(m=800, b=1, q: int or np.array = 1, is_visualized=True):

    # 生成具有不同峰度的随机数
    x = np.random.normal(0, 1, size=(m, 1))
    absolute_value_x = np.where(x > 0, 1, -1)
    bins = np.linspace(-4, 4, 100)
    for qi in q:
        xi = np.abs(x) ** qi * absolute_value_x
        plt.hist(xi, bins=bins, alpha=0.3, label=f'q={round(qi, 3)}')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 10}, frameon=False)
        plt.ylim(0, 60)
    plt.title("sub2super-Gaussian_dist")
    plt.savefig("sub2super-Gaussian_dist.jpg", dpi=200)
    plt.show()

q = np.arange(0.5, 2.001, 0.3)

create_simulated_data(q=q)

# # 定义形状参数
# beta_values = np.linspace(1, 3, 30)
#
# # 生成具有不同峰度的随机数
# random_numbers = [stats.gennorm.rvs(beta, loc=0, scale=1, size=1000, random_state=42) for beta in beta_values]
# plt.figure(figsize=(10, 9))
# # 创建直方图
# for i, numbers in enumerate(random_numbers):
#     # print(len(numbers))# 定义箱子边缘
#     bins = np.linspace(-8, 8, 50)
#     plt.hist(numbers, bins=bins, alpha=0.3, label=f'beta={round(beta_values[i], 2)}')
#
# # 添加图例和标题
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 10}, frameon=True)
# plt.title('Histogram of Random Numbers with Different Kurtosis')
#
# # 显示图形
# plt.show()

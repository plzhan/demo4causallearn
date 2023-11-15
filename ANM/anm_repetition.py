# # -*- coding:utf-8 -*-
# # Author:        zhanpl
# # Product_name:  PyCharm
# # File_name:     anm_repetition
# # @Time:         18:58  2023/11/2
# import warnings
# warnings.filterwarnings("ignore")
# from copy import deepcopy
# from tqdm import tqdm
# import threading
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel as C
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# from sklearn.model_selection import train_test_split
# from Experiments.HSIC import hsic_gam
# # create datas
# def create_simulated_data(m=300, b=1, q=1, is_visualized=True):
#     def f(x):
#
#         y = x + b * x ** 3
#         # y = b * np.exp(x)
#         return y
#
#     # 定义形状参数
#     # beta_values = np.linspace(1, 3, 30)
#     q += 1
#
#     # 生成具有不同峰度的随机数
#     # random_nums4x = [stats.gennorm.rvs(beta, loc=0, scale=1, size=1000, random_state=42) for beta in beta_values]
#     # random_nums4x = stats.gennorm.rvs(q, loc=0, scale=1, size=1000, random_state=42)
#     x = np.random.normal(0, 1, size=(m, 1))
#     n = np.random.normal(0, 1, size=(m, 1))
#     absolute_value_x = np.where(x > 0, 1, -1)
#     absolute_value_n = np.where(n > 0, 1, -1)
#     x = np.abs(x) ** q
#     n = np.abs(n) ** q
#     # print((x<0).sum())
#     x = x * absolute_value_x
#     n = n * absolute_value_n
#     # print((x<0).sum())
#     data_set = [x, f(x) + n]
#
#     x_min, x_max = np.min(x), np.max(x)
#     line_x = np.arange(x_min, x_max, 0.01)
#     line_y = f(line_x)
#     if is_visualized:
#         plt.plot(line_x, line_y, c='black')
#         plt.scatter(data_set[0], data_set[1], s=3, c='black')
#         plt.show()
#     # return
#     return data_set
#
#
# def get_an_estimate_model(data, is_split=True, to_ward="forward", is_visualized=True):
#     kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
#     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
#     if is_split:
#         X_train, X_test, y_train, y_test = train_test_split(data[:, :1], data[:, 1:], shuffle=True, test_size=.3, random_state=42)
#     else:
#         X_train, y_train = data[:, :1], data[:, 1:]
#     # fit model
#     gp.fit(X_train, y_train)
#
#     sorted_indices = X_train[:, 0].argsort()
#     X_train = X_train[sorted_indices]
#     y_train = y_train[sorted_indices]
#
#     # 使用训练好的模型进行预测
#     X = X_test if is_split else X_train  # inference
#     Y = y_test if is_split else y_train  # inference
#     y_pred, sigma = gp.predict(X, return_std=True)
#
#     sorted_indices = X[:, 0].argsort()
#     X = X[sorted_indices]
#     y_pred = y_pred[sorted_indices]
#     Y = Y[sorted_indices]
#
#     # visualization
#     if is_visualized:
#         plt.figure()
#         plt.scatter(X, Y, c='k', label='data')
#         plt.plot(X, y_pred, 'r', label='prediction')
#         plt.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
#                          alpha=0.2, color='r')
#         plt.xlabel('X')
#         plt.ylabel('y')
#         plt.title(f'Gaussian Process Regression ({to_ward})')
#         plt.legend()
#         plt.show()
#     # 3, calculate the corresponding residual n_hat = y - f_hat(x)
#     n_hat = Y - y_pred[:, None]
#     return n_hat, X
#
# def ANM(data_, is_split=False, is_visualized=True):
#     """
#     this function is used as a model(ANM) to discovery the causality between x and y
#     1, test whether x and y are statistically independent if not
#     2, test whether a model is consistent with data by non-linear regression of y on x, get an estimate model f(·)
#     3, calculate the corresponding residual n_hat = y - f_hat(x)
#     4, test whether n_hat is dependent with x if so, accept; if not
#     5, test whether the reverse model fits the data
#
#     situation:
#     1, independent
#     2, both direction (x independent with n and so is y)
#     3, only one direction (only x independent with n)
#     4, neither one is consistent (x and y are dependent with n)
#     :return:
#     """
#     data = deepcopy(np.array(data_)[:, :, 0].T)
#     data1 = deepcopy(np.array(data_)[[1, 0], :, 0].T)
#     result = ""
#     # print("data_shape, ", data.shape)
#     # print("data1_shape, ", data1.shape)
#     # 1, test whether x and y are statistically independent (kernel methods) if not
#     is_independent = hsic_gam(data[:,  1:], data[:, 1:])
#     if is_independent:
#         result = "X is independent with Y"
#         # print(result)
#         return result
#
#     # 2, test whether a model is consistent with data by non-linear regression of y on x, get an estimate model f(·)
#     # (Gaussian Processes regressor)
#     n_hat1, X = get_an_estimate_model(data, is_split, is_visualized=is_visualized)
#     n_hat2, Y = get_an_estimate_model(data1, is_split, "backward", is_visualized=is_visualized)
#
#     # 3, test whether n_hat is independent with x if so, accept; if not
#     is_independent1 = hsic_gam(X, n_hat1)
#     if is_independent1:
#         result += " forward: | X -> Y |"
#
#     # 4, test whether the reverse model fits the data
#     is_independent2 = hsic_gam(Y, n_hat2)
#     if is_independent2:
#         result += " backward: | X <- Y |"
#     print(result)
#     return result
#
# # The first panel
# is_visualized = False
# b = 0
# nums = np.arange(0.5, 1.1, 0.05)
# proportion = []
# repetition_num = 2
# for q in tqdm(nums):
#     q = round(q, 3)
#     times = 1
#     forward = 0
#     backward = 0
#     while times <= repetition_num:
#         dataset = create_simulated_data(b=b, q=q, is_visualized=is_visualized)
#         result = ANM(dataset, is_split=True, is_visualized=is_visualized)
#         if "forward" in result:
#             forward += 1
#         if "backward" in result:
#             backward += 1
#         times += 1
#     proportion.append([forward, backward])
#
# plt.figure()
# plt.plot(nums, np.array(proportion)[:, :1], 'k', label='correct')
# plt.plot(nums, np.array(proportion)[:, 1:], 'r', label='reverse')
# plt.xlabel('q')
# plt.ylabel('$p_{accept}$')
# plt.title(f'b = 0')
# plt.legend()
# plt.show()
# plt.savefig('b0_q05-20.jpg', dpi=200)
# # print("The result is:", result)
#
# # The second panel
#

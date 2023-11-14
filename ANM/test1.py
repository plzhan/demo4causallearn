# -*- coding:utf-8 -*-
# Author:        zhanpl
# Product_name:  PyCharm
# File_name:     test1
# @Time:         22:30  2023/11/11
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def create_simulated_data(m=300, b=1, q=1):
    def f(x):

        y = x + b * x ** 3
        return y

    # 定义形状参数
    # beta_values = np.linspace(1, 3, 30)
    q += 1

    # 生成具有不同峰度的随机数
    # random_nums4x = [stats.gennorm.rvs(beta, loc=0, scale=1, size=1000, random_state=42) for beta in beta_values]
    # random_nums4x = stats.gennorm.rvs(q, loc=0, scale=1, size=1000, random_state=42)
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

    plt.plot(line_x, line_y, c='black')
    plt.scatter(data_set[0], data_set[1], s=3, c='black')
    plt.show()
    # return
    return data_set

# 生成标准正态分布数据
X, y_true = create_simulated_data()
X_train, X_test, y_train, y_test = train_test_split(X, y_true, shuffle=True, test_size=.3,
                                                    random_state=42)

# X = np.random.normal(0, 1, (100, 1)) # 输入数据
y = 1 * X ** 3 + X  # 通过三次函数拟合得到的输出数据
# y = y_true + np.random.normal(0, 1, (100, 1))  # 添加高斯噪声的输出数据

# 选择核函数
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(1e-1)

# 构建高斯过程模型
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合模型
model.fit(X, y_true)

# 预测结果
x_pred = np.linspace(-20, 20, 1000)[:, np.newaxis]
y_pred, sigma = model.predict(x_pred, return_std=True)

# 绘制结果
plt.figure()
plt.scatter(X, y, c='k', label='Noisy Data', s=8)
# plt.plot(X, y_true, 'r', label='True Function')
plt.plot(x_pred, y_pred, 'g', label='Predicted Function')
plt.fill_between(x_pred.flatten(), (y_pred - 1.96 * sigma).flatten(), (y_pred + 1.96 * sigma).flatten(), alpha=0.1, color='g')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

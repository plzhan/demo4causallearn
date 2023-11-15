# # -*- coding:utf-8 -*-
# # Author:        zhanpl
# # Product_name:  PyCharm
# # File_name:     anm_repetition
# # @Time:         18:58  2023/11/2
# import math
# import numpy as np
# from copy import deepcopy
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel as C
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.model_selection import train_test_split
# from Experiments.HSIC import hsic_gam
# import torch
# import gpytorch
# from matplotlib import pyplot as plt
# torch.device("cuda:0")
#
# # import matplotlib
# # matplotlib.use('TkAgg')
#
# # create datas
# def create_simulated_data(m=300, b=1, q=1):
#     def f(x):
#
#         y = torch.add(x, b*x**3).cuda()
#         return y
#
#     # 定义形状参数
#     # beta_values = np.linspace(1, 3, 30)
#     q += 1
#
#     # 生成具有不同峰度的随机数
#     # random_nums4x = [stats.gennorm.rvs(beta, loc=0, scale=1, size=1000, random_state=42) for beta in beta_values]
#     # random_nums4x = stats.gennorm.rvs(q, loc=0, scale=1, size=1000, random_state=42)
#     x = torch.normal(0, 1, size=(m, 1)).cuda()
#     n = torch.normal(0, 1, size=(m, 1)).cuda()
#     absolute_value_x = torch.where(x > 0, 1, -1)
#     absolute_value_n = torch.where(n > 0, 1, -1)
#     x = torch.abs(x) ** q
#     n = torch.abs(n) ** q
#     # print((x<0).sum())
#     x = x * absolute_value_x
#     n = n * absolute_value_n
#     # print((x<0).sum())
#     data_set = [x, f(x) + n]
#
#     x_min, x_max = torch.min(x).to(torch.int16), torch.max(x).to(torch.int16)
#     line_x = torch.arange(x_min, x_max, 0.01)
#     line_y = f(line_x)
#
#     plt.plot(line_x.cpu(), line_y.cpu(), c='black')
#     plt.scatter(data_set[0].cpu(), data_set[1].cpu(), s=3, c='black')
#     plt.show()
#     # return
#     return data_set
#
#
# # We will use the simplest form of GP model, exact inference
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#
# def get_an_estimate_model(data, is_split=True, to_ward="forward"):
#     kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-5, 1e3))
#     gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
#     if is_split:
#         X_train, X_test, y_train, y_test = train_test_split(data[:, :1], data[:, 1:], shuffle=True, test_size=.3,
#                                                             random_state=42)
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
#     plt.figure()
#     plt.scatter(X, Y, c='k', label='data')
#     plt.plot(X, y_pred, 'r', label='prediction')
#     plt.fill_between(X.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
#                      alpha=0.3, color='r')
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.title(f'Gaussian Process Regression ({to_ward})')
#     plt.legend()
#     plt.show()
#     # 3, calculate the corresponding residual n_hat = y - f_hat(x)
#     n_hat = Y - y_pred[:, None]
#     return n_hat, X
#
#
# def ANM(data_, is_split=False):
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
#     result = "result: "
#     print("data_shape, ", data.shape)
#     print("data1_shape, ", data1.shape)
#     # 1, test whether x and y are statistically independent (kernel methods) if not
#     is_independent = hsic_gam(data[:, 1:], data[:, 1:])
#     if is_independent:
#         result = "X is independent with Y"
#         print(result)
#         return result
#
#     # 2, test whether a model is consistent with data by non-linear regression of y on x, get an estimate model f(·)
#     # (Gaussian Processes regressor)
#     n_hat1, X = get_an_estimate_model(data, is_split)
#     n_hat2, Y = get_an_estimate_model(data1, is_split, "backward")
#
#     # 3, test whether n_hat is independent with x if so, accept; if not
#     is_independent1 = hsic_gam(X, n_hat1)
#     if is_independent1:
#         result += "| X -> Y |"
#
#     # 4, test whether the reverse model fits the data
#     is_independent2 = hsic_gam(Y, n_hat2)
#     if is_independent2:
#         result += "| X <- Y |"
#     return result,
#
#
# # initialize likelihood and model
# dataset = create_simulated_data(m=500, b=0, q=2)
# train_x, test_x, train_y, test_y = train_test_split(dataset[0], dataset[1], test_size=.3, random_state=42, shuffle=True)
# train_x, test_x, train_y, test_y = \
#     torch.Tensor(train_x).clone().detach().requires_grad_(True),\
#     torch.Tensor(test_x).clone().detach().requires_grad_(True),\
#     torch.Tensor(train_y).clone().detach().requires_grad_(True),\
#     torch.Tensor(test_y).clone().detach().requires_grad_(True)
#
# likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
# model = ExactGPModel(train_x, train_y, likelihood).cuda()
#
# import os
#
# smoke_test = ('CI' in os.environ)
# training_iter = 2 if smoke_test else 50
#
# # Find optimal model hyperparameters
# model.train()
# likelihood.train()
#
# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
#
# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#
# for i in range(training_iter):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     output = model(train_x)
#     # Calc loss and backprop gradients
#     loss = -mll(output, train_y)
#     loss.sum().backward()
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         i + 1,
#         training_iter,
#         loss.sum().item(),
#         model.covar_module.base_kernel.lengthscale.item(),
#         model.likelihood.noise.item()
#     ))
#     optimizer.step()
#
# # result = ANM(dataset, is_split=True)
# # print("The result is:", result)
# # f_preds = model(test_x)
# # y_preds = likelihood(model(test_x))
# #
# # f_mean = f_preds.mean
# # f_var = f_preds.variance
# # f_covar = f_preds.covariance_matrix
# # f_samples = f_preds.sample(sample_shape=torch.Size(1000, ))
#
# # Get into evaluation (predictive posterior) mode
# model.eval()
# likelihood.eval()
#
# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))
#
# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))
#
#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])

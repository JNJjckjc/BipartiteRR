import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import copy
import random

from sklearn.model_selection import train_test_split, KFold
from sklearn import datasets

# from self_tool import metrics_science

test_X = pd.DataFrame([[4, 00, 3],
                       [2, 14, 2],
                       [2, 47, 2],
                       [3, 35, 2],
                       [8, 47, 3],
                       [5, 29, 2],
                       [6, 81, 3],
                       [1, 23, 3],
                       [5, 36, 2],
                       [3, 82, 3]], columns=['a', 'b', 'c'])

# map_test_X = pd.DataFrame([[3, 47, 3],
#                            [2, 14, 2],
#                            [2, 14, 2],
#                            [3, 35, 2],
#                            [3, 47, 3],
#                            [3, 14, 2],
#                            [1, 47, 3],
#                            [1, 35, 3],
#                            [1, 47, 2],
#                            [1, 35, 3]], columns=['a', 'b', 'c'])

test_y = pd.DataFrame([[6],
                       [2],
                       [9],
                       [3],
                       [5],
                       [8],
                       [2],
                       [8],
                       [3],
                       [1]], columns=['d'])
pass


pass # 多进程
# if __name__ == '__main__':
#     def test_mutpro1(j):
#         for i in range(100000):
#             print('i: {}'.format(i*j))
#
#     def test_mutpro2(i):
#         for j in range(100000):
#             print('j: {}'.format(i*j))
#
#     from multiprocessing import Process
#
#     p1 = Process(target=test_mutpro1(10))
#     p1.start()
#     p2 = Process(target=test_mutpro2(3))
#     p2.start()

pass # 求xgb mean
# xgb_all_rate = [[[13.116042303251215, 10.28786896474573, 7.535232106361098, 7.085879289274224, 6.70920100223407],
#                  [13.55479096678323, 8.971425301589097, 7.156101770339508, 6.602156750023871, 6.905949119336024],
#                  [13.186893951079373, 8.942202368600508, 7.955460083746305, 6.817871867781809, 6.702263043563869],
#                  [21.85876809629537, 10.539466739525727, 7.730985186303792, 7.002392830611763, 6.798357636265402],
#                  [21.164046864039925, 11.761509261032941, 7.9281868717450354, 7.504508422747641, 6.444950196824884]],
#
#                 [[6.376928542883634, 4.801565464110829, 4.593716472205708, 4.072894335156714, 4.1460846885410145],
#                  [5.981777906499257, 4.64027479677645, 4.459759475357332, 4.225926341578874, 4.089430017021225],
#                  [6.609870587218242, 4.475591046035677, 4.38380638795126, 4.1534823773891425, 4.27719522003869],
#                  [8.45156239120569, 4.897209342052408, 4.495867578819382, 4.312686131433068, 4.16795249182088],
#                  [7.907059710758562, 6.055449720060725, 4.410321133074033, 4.597766119303088, 4.193623874065716]]]
#
# mean=[]
# for i in range(len(xgb_all_rate)):
#     for j in range(len(xgb_all_rate[i])):
#         mean.append(np.mean(xgb_all_rate[i][j]))
#     print(mean)
#     mean=[]

# import datetime
# cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# print(cur_time)

# 时间戳
# import time
# cur_time1 = int(time.time())
# a = 0
# for i in range(100000000):
#     a += 1
# cur_time2 = int(time.time())
# print(cur_time2, cur_time2)
# print(cur_time2-cur_time1)


# from sklearn.datasets import make_regression
# path_x = 'F:/XGBoost_science/temporary_data/data_x'
# path_y = 'F:/XGBoost_science/temporary_data/data_y'
# sk_data_x, sk_data_y = make_regression(1000, 4, random_state=1)
# data_x = pd.DataFrame(sk_data_x)
# data_y = pd.DataFrame(sk_data_y)
# metrics_science.save_sklearn_data(path_x, data_x)
# metrics_science.save_sklearn_data(path_y, data_y)

pass
# a = np.sum(np.power(test_y - np.mean(test_y), 2))
# print(a[0])
# print(np.sum(np.power(test_y - np.mean(test_y), 2)).values[0])
# print(np.sum(test_y)[0])

# kFold_data_list = []
# for train_index, test_index in KFold(n_splits=3).split(test_X):
#     # print(train_index, test_index)
#     X_train, X_test = test_X.iloc[train_index, :], test_X.iloc[test_index, :]
#     y_train, y_test = test_y.iloc[train_index, :], test_y.iloc[test_index, :]
#     kFold_data_list.append([X_train, X_test, y_train, y_test])
#     # print(X_train, '\n', X_test, '\n', y_train, '\n', y_test)
# print(kFold_data_list)
# print(kFold_data_list[0][3].iloc[0, 0])
# for k in range(len(kFold_data_list)):
#     X_train, X_test, y_train, y_test = kFold_data_list[k][0], kFold_data_list[k][1], kFold_data_list[k][2], kFold_data_list[k][3]
#     print(X_train, '\n', X_test, '\n', y_train, '\n', y_test)

# max = 0.5
# min = 2
# result = []
# for n in range(1, 1000000):
#     f1 = n*n
#     f2 = (n+0.1)
#     f3 = np.power(n-1, 2)
#     f4 = (n+1.1)
#     fn = f1/f2 - f3/f4
#     if fn > max:
#         max = fn
#     if fn < min:
#         min = fn
#     result.append(fn)
# plt.plot(range(1, 1000000), result)
# plt.show()
# print(max)
# print(min)

# X, y = datasets.make_classification(n_samples=10,
#                                     n_features=5,
#                                     random_state=1)
# X = pd.DataFrame(X)
# y = pd.DataFrame(y)
# print(X)
# print(y)

# 0 -0.191836  1.054923 -0.729076 -1.146514  1.446343
# 1 -1.117310  0.794953  3.116518 -2.859616 -1.526374

# test_ = pd.DataFrame([ ['ab'],
#                        ['cd'],
#                        ['ab'],
#                        ['a'],
#                        ['cd']], columns=['d'])
# test_ = metrics_science.y_label_mapping(test_, binary_classification=True)
# print(test_)
# from sklearn.metrics import accuracy_score
# a = [-1, 1, -1, 1, 1]
# b = [-1, 1, -1, -1, 1]
# print(accuracy_score(a, b))
# print(1-accuracy_score(a, b))

# test_X_list = random.sample(range(test_X.shape[0]), test_X.shape[0])
# test_X_1 = pd.DataFrame(test_X, index=test_X_list)
# print(test_X_1)
# test_X_1.drop([1, 2, 3], inplace=True)
# print(test_X_1)

# l_a = range(10)
# l_u = [3, 6, 1, 8, 4, 9, 2]
# l_d = [9, 3, 8]
# print([j for j in l_u if j not in l_d])
# l_s = [i for i in l_a if i not in [j for j in l_u if j not in l_d]]
# print(l_s)

# ************************************************************ 函数释放类测试，链式取值，random_state参数,
# class A:
#     def pri(self, i=1):
#         i += 1
#         print(i)
#     def fit(self, j=1):
#         j += 2
#         print(j)
#
# def all_proce(self_c=None):
#     self_o = self_c()
#     self_o.pri()
#     self_o.fit()
#
# for k in [1, 2]:
#     all_proce(self_c= A)
# SettingWithCopyException: 链式取值
# for i in range(test_y.shape[0]):
#     test_y.iloc[i, 0] = 2 * test_y.iloc[i, 0] + 1
# train_test_split random_state参数固定，返回值索引固定
# print(test_y)
# X_train, X_test, y_train, y_test = train_test_split(test_X, test_y, train_size=0.7, random_state=1)
# print(X_train)
# print(X_test)
# X_train1, X_test1, y_train1, y_test1 = train_test_split(map_test_X, test_y, train_size=0.7, random_state=1)
# print(X_train1)
# print(X_test1)

# ************************************************************ 连续特征值离散化：
# def continue_to_dispersed(dataset_X):
#     dataset_X = dataset_X.reset_index(drop=True)
#     for feature in dataset_X.columns:
#         three_to_one_list = []
#         feature_value_list = list(np.unique(dataset_X[feature]))
#         if len(feature_value_list) < 3:
#             continue
#         for index in range(0, len(feature_value_list), 3):
#             if index < int(len(feature_value_list)/3) * 3 :
#                 three_to_one_list.append((feature_value_list[index] + feature_value_list[index + 1] +
#                 feature_value_list[index + 2]) / 3)
#             else:
#                 if len(feature_value_list) % 3 == 1:
#                     three_to_one_list.append(feature_value_list[-1])
#                 if len(feature_value_list) % 3 == 2:
#                     three_to_one_list.append((feature_value_list[-1] + feature_value_list[-2]) / 2)
#
#         for value_index in range(dataset_X.shape[0]):
#             dataset_X.loc[value_index, feature] = three_to_one_list[int(feature_value_list.index(
#             dataset_X.loc[value_index, feature]) / 3)]
#
#     return dataset_X
#
# print(test_X)
# test_X = continue_to_dispersed(test_X)
# print(test_X)

# ************************************************************ 将数据加载或写入文件
# with open('../data/dataset_test_result/test.txt', 'r') as f:
#     str = list(f.read())
# print(str)

# 将实验数据写入txt文档
# list = [3, 6, 7]
# with open('../data/dataset_test_result/test_delete.txt', 'a') as f:
#     f.write(str(list) + '\r\n')

# with open('../data/dataset_test_result/test.txt', 'a') as f:
#     for i in list:
#         f.write(str(i) + ',')
#     f.write('\r\n')
# MSE = [3, 6]
# MSE.insert(0, os.path.basename(__file__))
# print(MSE)
# with open('../data/test_data.txt', 'a') as f:
#     f.write(str(MSE) + '\r\n')
# del MSE[0]
# print(MSE)

# ************************************************************ 残差
# y = pd.DataFrame([4, 6, 7])
# residual_pred = np.zeros((3, 1))
# print(y - residual_pred)

# x = pd.read_csv('../data/abalone/new_abalone_X')
# self_metrics.data_bar(x)

# y = pd.read_csv('../data/abalone/new_abalone_y')
# print(y)
# for i in range(y.shape[0]):
#     y.iloc[i][0] = 14.5 * (y.iloc[i][0] + 1)
# print(y)
# y.to_csv('../data/abalone/re_mapping_abalone_y', index=False)

# ************************************************************ abalone
# y = pd.read_csv('../data/abalone/abalone_y')
# print('np.mean(y): {}'.format(np.mean(y)))
# print('y - np.mean(y): {}'.format(y - np.mean(y)))
# print('np.power(y - np.mean(y), 2): {}'.format(np.power(y - np.mean(y), 2)))
# a = np.sum(np.power(y - np.mean(y), 2))
# print(a.values[0])

# ************************************************************ 看 laplace取值范围
# true = 0
# false = 0
# for i in range(90):
#     if np.random.laplace(0, 0.8) > np.random.laplace(0, 5):
#         true += 1
#     else:
#         false += 1
# print(true, false)
#
# x = []
# y = 0
# for j in range(1, 11):
#     for i in range(99):
#         if np.random.laplace(0, 0.99 * 2 / j) >= 2 or np.random.laplace(0, 0.99 * 2 / j) <= -2:
#             y += 1
#     x.append(y/99)
# plt.plot(range(1, 11), x)
# plt.show()

# ************************************************************ .loc drop
# y = test_y
# print(y.loc[2, y.columns].values[0])
# for m in range(5):
#     if abs(y.loc[m, 'a']) > 5:
#         y.drop([m], inplace=True)
#         print(y)
# print(y)
# print(np.power(y - np.mean(y), 2))
# print(np.sum(np.power(y - np.mean(y), 2)).values[0])

# ************************************************************ 用代码的形式展示敏感度范围
# mse_jian = []
# max_value = 0
# max_value_index = 0
# x = range(1000000)
# for n in x :
#     n = n+1
#     # value = (3*n*n - 0.8*n - 0.1) / (n*n + 1.2*n + 0.11) # 前半部分：左子树
#     value = abs(((n*n) / (n+0.1)) - (((n-1)*(n-1)) / (n+1.1))) # 后半部分：总样本
#     if value > max_value:
#         # if value <= 3:
#         max_value = value
#         max_value_index = n
#     mse_jian.append(value)
#     # if value > 3:
#     #     mse_jian[-1] = mse_jian[-2]
#
# plt.plot(x, mse_jian, 'r*-')
# plt.xlabel('样本数')
# plt.ylabel('敏感度值')
# plt.show()
# print('max_value: {},  max_value_index: {}'.format(max_value, max_value_index))

# ************************************************************ 这种循环直接生成对象导致建的对象几乎都在同一个位置
# from science_research import XGBoost
# list = []
# for i in range(50):
#     xgb = XGBoost.xgbRegression(n_estimator= i + 5)
#     list.append(id(xgb))
# xgb1 = XGBoost.xgbRegression(n_estimator= 5)
# xgb2 = XGBoost.xgbRegression(n_estimator= 5)
# xgb3 = XGBoost.xgbRegression(n_estimator= 5)
# xgb4 = XGBoost.xgbRegression(n_estimator= 5)
# xgb5 = XGBoost.xgbRegression(n_estimator= 5)
# list.append(id(xgb1))
# list.append(id(xgb2))
# list.append(id(xgb3))
# list.append(id(xgb4))
# list.append(id(xgb5))
# uni = len(np.unique(list))
# print(list)
# print(uni)

# ************************************************************ lapulasi 测试
# 隐私预算从 0.5 取到 5 时laplace取1000次的平均值：[0.05551630688927349, 0.025266355765233908, -0.03055327755574232, -0.01592110843325335, 0.0062091483895703806, 0.02063309036186338, -0.01841566191774147, 0.0024062794972235764, 0.016786197599202385, -0.004598696598312236]
# 隐私预算从 0.5 取到 5 时laplace绝对值超过 2 的比率：[0.318, 0.1, 0.043, 0.011, 0.007, 0.002, 0.001, 0.0, 0.0, 0.0]
# 隐私预算从 0.5 取到 5 时laplace取值小于 0 的比率：[0.506, 0.485, 0.498, 0.512, 0.503, 0.503, 0.51, 0.504, 0.468, 0.503]
# lapalace_value = 0
# count_over_two = 0
# count_low_zero = 0
# list_mean = []
# list_exp = []
# over_two_rate = []
# low_zero_rate = []
# for exp_budget in range(1, 11):
#     for i in range(1000):
#         lapalace_value = np.random.laplace(0, 0.91 * 2/exp_budget)
#         list_exp.append(lapalace_value)
#         if abs(lapalace_value) > 2:
#             count_over_two += 1
#         if lapalace_value < 0:
#             count_low_zero += 1
#     list_mean.append(np.mean(list_exp))
#     over_two_rate.append(count_over_two/1000)
#     low_zero_rate.append(count_low_zero/1000)
#     list_exp = []
#     count_over_two = 0
#     count_low_zero = 0
#
# print('隐私预算从 0.5 取到 5 时laplace取1000次的平均值：{}'.format(list_mean))
# print('隐私预算从 0.5 取到 5 时laplace绝对值超过 2 的比率：{}'.format(over_two_rate))
# print('隐私预算从 0.5 取到 5 时laplace取值小于 0 的比率：{}'.format(low_zero_rate))

# ************************************************************ xgb 敏感度的代码测试
# value_jia_jia = []
# value_jia_jian = []
# value_jian_jia = []
# value_jian_jian = []
# max_value_jia_jia = []
# max_value_jia_jian = []
# max_value_jian_jia = []
# max_value_jian_jian = []
# n_samples = [100, 1000, 5000, 10000, 20000]
# for n in n_samples:
#     for nl in range(n):
#         nl = nl + 1
#         value_jia_jia.append( abs( np.power(nl+1, 2)/(nl+1.1) - np.power(nl, 2)/(nl+0.1) + np.power(n, 2)/(n+0.1) - np.power(n+1, 2)/(n+1.1) ) )
#         value_jia_jian.append( abs( np.power(nl+1, 2)/(nl+1.1) - np.power(nl, 2)/(nl+0.1) + np.power(n, 2)/(n+0.1) - np.power(n-1, 2)/(n+1.1) ) )
#         value_jian_jia.append( abs( np.power(nl-1, 2)/(nl+1.1) - np.power(nl, 2)/(nl+0.1) + np.power(n, 2)/(n+0.1) - np.power(n+1, 2)/(n+1.1) ) )
#         value_jian_jian.append( abs( np.power(nl-1, 2)/(nl+1.1) - np.power(nl, 2)/(nl+0.1) + np.power(n, 2)/(n+0.1) - np.power(n-1, 2)/(n+1.1) ) )
#     max_value_jia_jia.append(max(value_jia_jia))
#     max_value_jia_jian.append(max(value_jia_jian))
#     max_value_jian_jia.append(max(value_jian_jia))
#     max_value_jian_jian.append(max(value_jian_jian))
#     value_jia_jia = []
#     value_jia_jian = []
#     value_jian_jia = []
#     value_jian_jian = []
#
# plt.plot(n_samples, max_value_jia_jia, 'r*-', label = 'jia_jia')
# plt.plot(n_samples, max_value_jia_jian, 'y>-', label = 'jia_jian')
# plt.plot(n_samples, max_value_jian_jia, 'b^-', label = 'jian_jia')
# plt.plot(n_samples, max_value_jian_jian, 'g<-', label = 'jian_jian')
# plt.xlabel('样本总数')
# plt.ylabel('最大敏感度')
# plt.legend()
# plt.show()

# ************************************************************ AAAI 衰减率
# x = pd.DataFrame([[1, 2],
#                  [2, 5]])
# print(x)
# x.drop([1], axis=1, inplace=True)
# print(x)

# X = pd.read_csv('F:/XGBoost/data/YearPredictionMSD/YearPredictionMSD', header=None)
# y = pd.DataFrame(X.iloc[:, 0])
# y.columns = ['Year']
# y.to_csv('F:/XGBoost/data/YearPredictionMSD/YearPredictionMSD_y', index=False)

# y = pd.read_csv('/data/YearPredictionMSD/new_YearPredictionMSD_y') 512
# metrics_science.continue_feature_value(y, ['Year'])
# shrinkage_list = np.zeros((10, 1))
# print(shrinkage_list)
# for s in range(10):
#     shrinkage_list[s] = 5 / shrinkage_list[s]
# print(shrinkage_list)
# print(np.mean(shrinkage_list))

# ************************************************************ scale 越大，Laplace噪声越大
# list_001 = []
# list_01 = []
# list_05 = []
# list_1 = []
# for i in range(10):
#     list_001.append(np.random.laplace(0, 0.01))
#     list_01.append(np.random.laplace(0, 0.1))
#     list_05.append(np.random.laplace(0, 0.5))
#     list_1.append(np.random.laplace(0, 1))
# list_001.append(np.mean(list_001))
# list_01.append(np.mean(list_01))
# list_05.append(np.mean(list_05))
# list_1.append(np.mean(list_1))
# print(list_001)                                   # 0.0003638866845764279  -0.004955404603999896
# print(list_01)                                    # -0.061822701385286186  0.049182399063166446
# print(list_05)                                    # 0.1400495066775483     0.07804258760533558
# print(list_1)                                     # 0.7149620139073958     -0.23024664188078345

# ************************************************************ 显示字体颜色
# print('\033[0;32m输出吃饭来玩么[feg]的基本\033[0m') ********************************************************************
# \033[0;32m
# \033[0m

# ************************************************************ 差分隐私并行训练，每棵树样本不同
# X = pd.read_csv('F:/XGBoost/data/abalone/new_abalone_X')
# y = pd.read_csv('F:/XGBoost/data/abalone/new_abalone_y')
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, random_state=1)
# X = X_train.reset_index(drop=True)
# y = y_train.reset_index(drop=True)
# print(X)
# print(y)
#
# samples = X.shape[0]
# n = 3
# for i in range(n):
#     choose_samples_list = random.sample(range(samples - i * int(samples/n)), int(samples/n))
#     X_new = pd.DataFrame(X, index=choose_samples_list).reset_index(drop=True)
#     y_new = pd.DataFrame(y, index=choose_samples_list).reset_index(drop=True)
#     X.drop(choose_samples_list, inplace=True)
#     y.drop(choose_samples_list, inplace=True)
#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)
#     if i == n-1:
#         X_new = X
#         y_new = y

# ************************************************************ np.zeros 类型测试
# predict_value = np.zeros((4, 1))
# print(type(predict_value))
# print(predict_value[1][0])
# del predict_value[1][0]
# print(predict_value)

# ************************************************************ 打乱数据集
# x = test_X
# index_list = random.sample(range(x.shape[0]), x.shape[0])
# x = pd.DataFrame(x, index=index_list)
# print(x)
# x = x.reset_index(drop=True)
# print(x)

# def upset_dataset(data): # 打乱数据集
#     copy_data = copy.deepcopy(data)
#     n_samples = copy_data.shape[0]
#     for i in range(10):
#         upset_index_list = random.sample(range(n_samples), n_samples)
#         copy_data = pd.DataFrame(copy_data, index=upset_index_list).reset_index(drop=True)
#     return copy_data

# x = upset_dataset(x)
# print(x)

# y_columns = x.columns
# y = pd.DataFrame(x[y_columns[0]], columns=[y_columns[0]])












































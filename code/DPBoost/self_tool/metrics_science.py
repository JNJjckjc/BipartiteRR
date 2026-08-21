a = '/home/user2/chenPycode/linux_python/vision2/'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copy
import sys
import os
import random
import time
import datetime

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import xgb_no_privacy
import XGBoost
import AAAI_paper
import GBDT
import XGBoost_g_brr
import XGBoost_g_lap
# 得到处理过的训练集和测试集
def get_train_test_data(file_X=None, file_y=None, k=None, b=None, train_rate=0.8, back_mapping=True, binary_classification=False, kFold=False):
    X = pd.read_csv(file_X)
    y = pd.read_csv(file_y)
    X_train, X_test, y_train, y_test = None, None, None, None
    kFold_data_list = []
    if kFold is True:
        for train_index, test_index in KFold(n_splits=5).split(X):
            X_train = X.iloc[train_index, :].reset_index(drop=True)
            X_test = X.iloc[test_index, :].reset_index(drop=True)
            y_train = y.iloc[train_index, :].reset_index(drop=True)
            y_test = y.iloc[test_index, :].reset_index(drop=True)
            kFold_data_list.append([X_train, X_test, y_train, y_test])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_rate, random_state=1)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    if binary_classification is True:
        if kFold is True:
            for k_index in range(len(kFold_data_list)):
                k_y_test = kFold_data_list[k_index][3]
                for i in range(k_y_test.shape[0]):
                    if k_y_test.iloc[i, 0] > 0:
                        k_y_test.iloc[i, 0] = 1
                    else:
                        k_y_test.iloc[i, 0] = -1
                kFold_data_list[k_index][3] = k_y_test
        else:
            for i in range(y_test.shape[0]):
                if y_test.iloc[i, 0] > 0:
                    y_test.iloc[i, 0] = 1
                else:
                    y_test.iloc[i, 0] = -1
    if back_mapping is True:
        if kFold is True:
            for k_index in range(len(kFold_data_list)):
                for i in range(kFold_data_list[k_index][3].shape[0]):
                    kFold_data_list[k_index][3].loc[i, y_test.columns] = k * kFold_data_list[k_index][3].loc[i, y_test.columns] + b
        else:
            for i in range(y_test.shape[0]):
                y_test.loc[i, y_test.columns] = k * y_test.loc[i, y_test.columns] + b
    if kFold is True:
        return kFold_data_list
    else:
        return [[X_train, X_test, y_train, y_test]]

# 总过程
def all_process(xgboost_g_lap = False, xgboost_g_brr = False, xgboost=False, AAAI=False, gbdt=False, X_train=None, X_test=None, y_train=None, y_test=None,
                pri_bud=1, internal_rate=0.5, lap1_rate=0.5, tree=50, k=None, b=None, binary_classification=False):
    """
    训练树
    :param xgboost: 自己的实验
    :param AAAI: 复现AAAI
    :param gbdt: 复现GBDT
    :param X_train: 训练集X
    :param X_test: 测试集X
    :param y_train: 训练集y
    :param y_test: 测试集y
    :param pri_bud: 总隐私预算
    :param internal_rate: 每棵树内部节点隐私预算占总隐私预算的比率
    :param lap1_rate: 叶子节点分配给laplace1 的隐私预算占当前叶子节点总隐私预算的比率
    :param tree: 总共训练多少棵树
    :param k: 预测结果进行反映射的系数
    :param b: 同 k
    :param binary_classification: 是否进行二分类，默认进行回归
    :return: 测试误差(二分类)/均方根误差(回归)，样本删除率
    """
    # if xgboost is AAAI:
    #     sys.exit('文件 {} 第 {} 行发生错误：当前应训练且只能训练一个模型'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))
    model = None
    del_samples_rate = None
    if xgboost_g_lap is True:
        time_xgb_pre = int(time.time())
        model = XGBoost_g_lap.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud, internal_rate, lap1_rate)
        time_xgb_aft = int(time.time())
        print('隐私保护xgboost一轮五折训练时间平均每棵树：{}'.format((time_xgb_aft - time_xgb_pre) / tree))
    if xgboost_g_brr is True:
        time_xgb_pre = int(time.time())
        model = XGBoost_g_brr.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud, internal_rate, lap1_rate)
        time_xgb_aft = int(time.time())
        print('隐私保护xgboost一轮五折训练时间平均每棵树：{}'.format((time_xgb_aft - time_xgb_pre) / tree))
    if xgboost is True:
        time_xgb_pre = int(time.time())
        model = XGBoost.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud, internal_rate, lap1_rate)
        time_xgb_aft = int(time.time())
        print('隐私保护xgboost一轮五折训练时间平均每棵树：{}'.format((time_xgb_aft-time_xgb_pre)/tree))
    if AAAI is True:
        time_aaai_pre = int(time.time())
        model = AAAI_paper.xgbRegression(n_estimator=tree)
        del_samples_rate = model.fit(X_train, y_train, pri_bud)
        time_aaai_aft = int(time.time())
        print('隐私保护aaai一轮五折训练时间平均每棵树：{}'.format((time_aaai_aft - time_aaai_pre)/tree))
    if gbdt is True:
        time_gbdt_pre = int(time.time())
        # model = GBDT.xgbRegression(n_estimator=tree)
        model = GBDT.GBDTRegression(n_estimator=tree)
        model.fit(X_train, y_train, pri_bud)
        time_gbdt_aft = int(time.time())
        print('隐私保护gbdt一轮五折训练时间平均每棵树：{}'.format((time_gbdt_aft - time_gbdt_pre)/tree))
    y_pred = model.predict(X_test)

    # 二分类
    if binary_classification is True:
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        error_rate = (1 - accuracy_score(y_pred, y_test)) * 100
        print('\033[0;32m隐私预算为 {} 时的测试误差：{}\033[0m'.format(pri_bud, error_rate))
        return error_rate, del_samples_rate

    # 回归
    for i in range(len(y_pred)):
        y_pred[i] = k * y_pred[i] + b
    rmse = mean_squared_error(y_pred, y_test) ** 0.5
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\033[0;32m{} 隐私预算为 {} 时的均方根误差：{}\033[0m'.format(cur_time, pri_bud, rmse))
    return rmse, del_samples_rate

# 不加差分隐私
def no_privacy_process(file_x=None, file_y=None, tree=50, binary_classification=False):
    four_data = get_train_test_data(file_x, file_y, back_mapping=False)
    if len(four_data) != 1:
        sys.exit('文件 {} 第 {} 行发生错误： 这里不使用交叉验证'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))
    X_train, X_test, y_train, y_test = four_data[0][0], four_data[0][1], four_data[0][2], four_data[0][3]
    time_no_pri_xgb_pre = int(time.time())
    xgb = xgb_no_privacy.xgbRegression(n_estimator=tree)
    xgb.fit(X_train, y_train)
    time_no_pri_xgb_aft = int(time.time())
    print('无隐私保护xgboost一轮五折训练时间平均每棵树：{}'.format((time_no_pri_xgb_aft - time_no_pri_xgb_pre)/tree))
    y_pred = xgb.predict(X_test)

    # 二分类
    if binary_classification is True:
        for i in range(len(y_pred)):
            if y_pred[i] > 0:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        error_rate = (1 - accuracy_score(y_pred, y_test)) * 100
        print('\033[0;32m不加差分隐私时的测试误差：{}\033[0m'.format(error_rate))
        return error_rate

    # 回归
    rmse = mean_squared_error(y_pred, y_test) ** 0.5
    print('\033[0;32m不加差分隐私时的均方根误差：{}\033[0m'.format(rmse))
    return rmse

# 将实验数据写入txt文档
def save_experiment_result(result_list=None, list_0_label=None, save_file=None, first_line=False, privacy_list=None):
    result_list.insert(0, list_0_label)
    with open(save_file, 'a') as f:
        if first_line is True and privacy_list is not None:
            privacy_list.insert(0, 'all privacy budget')
            f.write('*' * 60 + str(privacy_list) + '\r\n')
            del privacy_list[0]
        f.write(str(result_list) + '\r\n')
    del result_list[0]



def visual_experiment_result(x_value, y_value_xgb, y_value_AAAI, base_line_list, y_value_GBDT=None, title=None, binary_classification=False, del_rate=False):
    xticks_index = range(len(x_value))
    plt.plot(xticks_index, y_value_xgb, 'r*-', label ='DP_xgboost')
    plt.plot(xticks_index, y_value_AAAI, 'g^-', label ='DPBoost')
    if y_value_GBDT is not None:
        plt.plot(xticks_index, y_value_GBDT, 'b>-', label='DP_gbdt')
    if len(base_line_list) == len(x_value) :
        plt.plot(xticks_index, base_line_list, 'ko-', label='xgboost')
    if del_rate is True:
        plt.xlabel('internal_privacy_rate')
    else:
        plt.xlabel('privacy budget')

    if binary_classification is True:
        plt.ylabel('test error (%)')
    elif del_rate is True:
        plt.ylabel('delete rate (%)')
    else:
        plt.ylabel('RMSE')
    plt.xticks(xticks_index, x_value)
    plt.legend()
    plt.show()

# 配合下面的函数实现指数机制
def random_pick(score_list, probability_list):
    """
    以相应的概率输出该值
    """
    # global item
    x = np.random.uniform(0, 1)
    cumulative_probability = 0
    i = -1 # 待返回的评分列表的下标
    for item, item_probability in zip(score_list, probability_list): # 当zip的两个参数为空列表时，for循环不执行
        i += 1
        cumulative_probability += item_probability
        if x < cumulative_probability: return i
    return len(score_list) - 1

# 指数机制
def exp(score, exp_budget, sensitive):
    """
    score：list，待选择的列表
    exp_budget：隐私预算
    sensitive：敏感度
    """
    exponents = []
    for i in score:
        expo = 0.5 * i * exp_budget / sensitive
        exponents.append(np.exp(expo))

    # 缩放，防止 RuntimeWarning
    max = np.max(exponents)
    exponents = exponents / max
    sum = np.sum(exponents)

    for j in range(len(exponents)):
        exponents[j] = exponents[j] / sum
    return random_pick(score, exponents)

# 将数据集特征标签进行映射：回归为[-1, 1]；
def y_label_mapping(label_list, binary_classification=False):
    """
    labal_list: 数据集目标特征列
    返回：映射后标签
         原目标特征取值的最大值
         原目标特征取值的最小值
         反映射系数
         反映射常数
    """
    # copy一份数据集
    label_list = copy.deepcopy(label_list)


    # 回归映射
    max = label_list.iloc[0, 0]
    min = label_list.iloc[0, 0]
    for i in range(len(label_list)):
        if label_list.iloc[i, 0] > max:
            max = label_list.iloc[i, 0]
        if label_list.iloc[i, 0] < min:
            min = label_list.iloc[i, 0]
    for j in range(len(label_list)):
        label_list.iloc[j, 0] = (2 * label_list.iloc[j, 0] - max - min) / (max - min)
    return label_list, max, min, (max - min) / 2, (max + min) / 2

# 显示每个特征的分裂点个数
def data_bar(data):
    """
    data: 待可视化的数据集: DataFrame
    """
    feature_list = []
    unique_list = []
    for feature in data.columns:
        feature_list.append(feature)
        unique_list.append(len(np.unique(data[feature])))

    # 把每个特征的分裂点个数可视化：柱状图
    plt.bar(range(len(feature_list)), unique_list, tick_label=feature_list)
    plt.xlabel('feature')
    plt.ylabel('unique_num')
    plt.show()

# 数据集柱状图：行为特征，列为非重复元素个数
def draw_bar(data, title=None):
    """
    数据集可视化（行：特征；列：非重复元素个数）
    data: 数据集
    """
    feature_list = data.columns
    unique_value_number = []
    for feature in feature_list:
        unique_value_number.append(len(np.unique(data[feature])))

    # 可视化
    plt.bar(feature_list, unique_value_number)
    for x, y in zip(feature_list, unique_value_number):
        plt.text(x, y+0.05, y, ha='center', va='bottom')
    plt.xlabel('特征')
    plt.xticks(rotation=300)
    plt.ylabel('非重复元素个数')
    plt.title(title)
    plt.show()

# 将传入的数据集中待映射特征里的所有标签映射成数值型类别
def mapping_feature_value(y, mapping_feature_list):
    """
    :param y: 待映射数据集
    :param mapping_feature_list: 待映射特征列表
    :return: 映射后的数据集
    """
    if not isinstance(y, pd.DataFrame) or not isinstance(mapping_feature_list, list):
        sys.exit('文件 {} 第 {} 行发生错误： 传入 mapping_feature_value 函数的参数格式有问题, 数据集:DataFrame, 特征列表:list'
                 .format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

    y = y.reset_index(drop=True)

    for feature in mapping_feature_list:
        label = list(np.unique(y[feature]))
        for i in range(y.shape[0]):
            for item in label:
                if item == y.loc[i, feature]:
                    y.loc[i, feature] = label.index(item) + 1
                    break
    return y

# 删除一阶导中不满足条件对应的样本
def del_samples(X, y_copy, threshold):
    """
    X：样本集
    y：目标值
    threshold：不符样本的阈值
    return: 返回处理后的X、y、删除的索引列表
    """
    # 保证传入的数据集 X 中已经拼接了一阶导
    if 'grad' not in X.columns:
        sys.exit('文件 {} 第 {} 行发生错误： 请先把一阶导 "grad" 列拼接到X上'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

    # 深拷贝传入的 X、y_coy, 防止每棵树训练的数据集不断减少
    X = copy.deepcopy(X)
    y_pre = copy.deepcopy(y_copy)

    # 取第一列
    y_columns = y_pre.columns
    y = pd.DataFrame(y_pre[y_columns[0]], columns=[y_columns[0]])

    index_list = []

    # 删除不符样本并重置索引
    for i in X.index:
        if abs(X.loc[i, 'grad']) > threshold:
            index_list.append(i)
    X.drop(index_list, inplace=True)
    y.drop(index_list, inplace=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 返回不符样本的比例
    return X, y, index_list



# 连续特征值离散化：
def continue_to_dispersed(dataset_X):
    dataset_X = dataset_X.reset_index(drop=True)
    for feature in dataset_X.columns:
        three_to_one_list = []
        feature_value_list = list(np.unique(dataset_X[feature]))
        if len(feature_value_list) < 100:
            continue
        for index in range(0, len(feature_value_list), 3):
            if index < int(len(feature_value_list)/3) * 3 :
                three_to_one_list.append((feature_value_list[index] + feature_value_list[index + 1] + feature_value_list[index + 2]) / 3)
            else:
                if len(feature_value_list) % 3 == 1:
                    three_to_one_list.append(feature_value_list[-1])
                if len(feature_value_list) % 3 == 2:
                    three_to_one_list.append((feature_value_list[-1] + feature_value_list[-2]) / 2)

        for value_index in range(dataset_X.shape[0]):
            dataset_X.loc[value_index, feature] = three_to_one_list[int(feature_value_list.index(dataset_X.loc[value_index, feature]) / 3)]

    return dataset_X

# 实验 - 相当于main()
def run_experiment(core_str=None, xgb_g_lap=False, xgb_g_brr=False, xgb=False, AAAI=False, Gbdt=False, Base=False, draw_picture=False, save_test_result=False,
                   binary_cla=False, k=None, b=None, all_tree=50, internal_rate=0.1, kFold=False, pri_list=None):

    # 所有隐私预算
    # all_privacy = [1, 2, 4, 6, 8, 10]
    if pri_list is None:
        all_privacy = [0.5, 1, 2, 3, 4, 5]
    else:
        all_privacy = pri_list

    # 任务类型：回归(默认) or 二分类
    task_type = 'regression'
    label_y_half = ''
    if binary_cla is True:
        task_type = 'classification'
        # label_y_half = '_0.5'

    file_X_xgb = 'E:/XGBoost_science/data/' + task_type + '/' + core_str + '/map_' + core_str + '_X'
    file_y = 'E:/XGBoost_science/data/' + task_type + '/' + core_str + '/map_' + core_str + '_y' + label_y_half
    save_file = 'E:/XGBoost_science/data/dataset_test_result/' + task_type + '/' + core_str
    RESULT_xgb, RESULT_AAAI, RESULT_gbdt, BASE_value_list = [], [], [], []
    if xgb_g_lap is True:
        DEL_samples_rate_xgb = []
        X_y_train_test_xgb = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp = [], []
            for k_index in range(len(X_y_train_test_xgb)):
                X_train_xgb, X_test_xgb = X_y_train_test_xgb[k_index][0], X_y_train_test_xgb[k_index][1]
                y_train, y_test = X_y_train_test_xgb[k_index][2], X_y_train_test_xgb[k_index][3]
                result_xgb, del_sample_rate_xgb = all_process(xgboost_g_lap=True,
                                                              X_train=X_train_xgb,
                                                              X_test=X_test_xgb,
                                                              y_train=y_train,
                                                              y_test=y_test,
                                                              pri_bud=privacy_budget,
                                                              internal_rate=internal_rate,
                                                              tree=all_tree,
                                                              k=k, b=b,
                                                              binary_classification=binary_cla)
                RESULT_xgb_temp.append(result_xgb)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_xgb)
            RESULT_xgb.append(np.mean(RESULT_xgb_temp))
            DEL_samples_rate_xgb.append(np.mean(DEL_samples_rate_xgb_temp))
        print('\033[0;32mRESULT_xgb_g_lap: {}\033[0m'.format(RESULT_xgb))
        print('\033[0;32mDEL_samples_rate_xgb_g_lap: {}\033[0m'.format(DEL_samples_rate_xgb))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_xgb, list_0_label='xgb_RESULT', save_file=save_file,
                                   first_line=True, privacy_list=all_privacy)
            save_experiment_result(result_list=DEL_samples_rate_xgb, list_0_label='xgb_del_rate', save_file=save_file)
    if xgb_g_brr is True:
        DEL_samples_rate_xgb = []
        X_y_train_test_xgb = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp = [], []
            for k_index in range(len(X_y_train_test_xgb)):
                X_train_xgb, X_test_xgb = X_y_train_test_xgb[k_index][0], X_y_train_test_xgb[k_index][1]
                y_train, y_test = X_y_train_test_xgb[k_index][2], X_y_train_test_xgb[k_index][3]
                result_xgb, del_sample_rate_xgb = all_process(xgboost_g_brr=True,
                                                              X_train=X_train_xgb,
                                                              X_test=X_test_xgb,
                                                              y_train=y_train,
                                                              y_test=y_test,
                                                              pri_bud=privacy_budget,
                                                              internal_rate=internal_rate,
                                                              tree=all_tree,
                                                              k=k, b=b,
                                                              binary_classification=binary_cla)
                RESULT_xgb_temp.append(result_xgb)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_xgb)
            RESULT_xgb.append(np.mean(RESULT_xgb_temp))
            DEL_samples_rate_xgb.append(np.mean(DEL_samples_rate_xgb_temp))
        print('\033[0;32mRESULT_xgb_g_brr: {}\033[0m'.format(RESULT_xgb))
        print('\033[0;32mDEL_samples_rate_xgb_g_brr: {}\033[0m'.format(DEL_samples_rate_xgb))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_xgb, list_0_label='xgb_RESULT', save_file=save_file,
                                   first_line=True, privacy_list=all_privacy)
            save_experiment_result(result_list=DEL_samples_rate_xgb, list_0_label='xgb_del_rate', save_file=save_file)
    if xgb is True:
        DEL_samples_rate_xgb = []
        X_y_train_test_xgb = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp = [], []
            for k_index in range(len(X_y_train_test_xgb)):
                X_train_xgb, X_test_xgb = X_y_train_test_xgb[k_index][0], X_y_train_test_xgb[k_index][1]
                y_train, y_test = X_y_train_test_xgb[k_index][2], X_y_train_test_xgb[k_index][3]
                result_xgb, del_sample_rate_xgb = all_process(xgboost=True,
                                                              X_train=X_train_xgb,
                                                              X_test=X_test_xgb,
                                                              y_train=y_train,
                                                              y_test=y_test,
                                                              pri_bud=privacy_budget,
                                                              internal_rate=internal_rate,
                                                              tree=all_tree,
                                                              k=k, b=b,
                                                              binary_classification=binary_cla)
                RESULT_xgb_temp.append(result_xgb)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_xgb)
            RESULT_xgb.append(np.mean(RESULT_xgb_temp))
            DEL_samples_rate_xgb.append(np.mean(DEL_samples_rate_xgb_temp))
        print('\033[0;32mRESULT_xgb: {}\033[0m'.format(RESULT_xgb))
        print('\033[0;32mDEL_samples_rate_xgb: {}\033[0m'.format(DEL_samples_rate_xgb))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_xgb, list_0_label='xgb_RESULT', save_file=save_file, first_line=True, privacy_list=all_privacy)
            save_experiment_result(result_list=DEL_samples_rate_xgb, list_0_label='xgb_del_rate', save_file=save_file)

    if AAAI is True:
        file_X_AAAI = '/home/user2/chenPycode/XGBoost_science/data/' + task_type + '/' + core_str + '/' + core_str + '_X'
        DEL_samples_rate_AAAI = []
        X_y_train_test_AAAI = get_train_test_data(file_X=file_X_AAAI,
                                                  file_y=file_y,
                                                  k=k, b=b,
                                                  back_mapping=not binary_cla,
                                                  binary_classification=binary_cla,
                                                  kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_xgb_temp, DEL_samples_rate_xgb_temp = [], []
            for k_index in range(len(X_y_train_test_AAAI)):
                X_train_AAAI, X_test_AAAI = X_y_train_test_AAAI[k_index][0], X_y_train_test_AAAI[k_index][1]
                y_train, y_test = X_y_train_test_AAAI[k_index][2], X_y_train_test_AAAI[k_index][3]
                result_AAAI, del_sample_rate_AAAI = all_process(AAAI=True,
                                                                X_train=X_train_AAAI,
                                                                X_test=X_test_AAAI,
                                                                y_train=y_train,
                                                                y_test=y_test,
                                                                pri_bud=privacy_budget,
                                                                tree=all_tree,
                                                                k=k, b=b,
                                                                binary_classification=binary_cla)
                RESULT_xgb_temp.append(result_AAAI)
                DEL_samples_rate_xgb_temp.append(del_sample_rate_AAAI)
            RESULT_AAAI.append(np.mean(RESULT_xgb_temp))
            DEL_samples_rate_AAAI.append(np.mean(DEL_samples_rate_xgb_temp))
        print('\033[0;32mRESULT_AAAI: {}\033[0m'.format(RESULT_AAAI))
        print('\033[0;32mDEL_samples_rate_AAAI: {}\033[0m'.format(DEL_samples_rate_AAAI))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_AAAI, list_0_label='AAAI_RESULT', save_file=save_file)
            save_experiment_result(result_list=DEL_samples_rate_AAAI, list_0_label='AAAI_del_rate', save_file=save_file)

    if Gbdt is True:
        X_y_train_test_gbdt = get_train_test_data(file_X=file_X_xgb,
                                                 file_y=file_y,
                                                 k=k, b=b,
                                                 back_mapping=not binary_cla,
                                                 binary_classification=binary_cla,
                                                 kFold=kFold)
        for privacy_budget in all_privacy:
            RESULT_gbdt_temp = []
            for k_index in range(len(X_y_train_test_gbdt)):
                X_train_gbdt, X_test_gbdt = X_y_train_test_gbdt[k_index][0], X_y_train_test_gbdt[k_index][1]
                y_train, y_test = X_y_train_test_gbdt[k_index][2], X_y_train_test_gbdt[k_index][3]
                result_gbdt, del_rate_temp_ = all_process(gbdt=True,
                                                          X_train=X_train_gbdt,
                                                          X_test=X_test_gbdt,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          pri_bud=privacy_budget,
                                                          tree=all_tree,
                                                          k=k, b=b,
                                                          binary_classification=binary_cla)
                RESULT_gbdt_temp.append(result_gbdt)
            RESULT_gbdt.append(np.mean(RESULT_gbdt_temp))
        print('\033[0;32mRESULT__gbdt: {}\033[0m'.format(RESULT_gbdt))
        if save_test_result is True:
            save_experiment_result(result_list=RESULT_gbdt, list_0_label='gbdt_RESULT', save_file=save_file)

    if Base is True:
        y_map = ''
        if binary_cla is True:
            y_map = 'map_'
        file_no_map_y = '/home/user2/chenPycode/DPBoost/data/' + task_type + '/' + core_str + '/' + y_map + core_str + '_y'
        BASE_value_list.append(no_privacy_process(file_x=file_X_xgb,
                                                  file_y=file_no_map_y,
                                                  tree=all_tree,
                                                  binary_classification=binary_cla))
        if save_test_result is True:
            save_experiment_result(result_list=BASE_value_list, list_0_label='no_privacy', save_file=save_file)

        visual_experiment_result(x_value=all_privacy,
                                 y_value_xgb=RESULT_xgb,
                                 y_value_AAAI=RESULT_AAAI,
                                 base_line_list=BASE_value_list * len(all_privacy),
                                 title=core_str,
                                 binary_classification=binary_cla)











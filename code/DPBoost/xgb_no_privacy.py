import numpy as np
import pandas as pd
import sys
import os
import copy
import random

from self_tool import metrics_science

tree_depth = 5

class TreeNode():
    """树结点"""
    def __init__(self, feature_idx=None, feature_val=None, node_val=None, left_child=None, right_child=None):
        """
        feature_idx: 该结点对应的划分特征索引
        feature_val: 划分特征对应的值
        node_val: 该结点存储的值，只有叶结点才存储类别信息，回归树存储结点的平均值，分类树存储类别出现次数最多的类别
        left_child: 左子树
        right_child: 右子树
        """
        self._feature_idx = feature_idx
        self._feature_val = feature_val
        self._node_val = node_val
        self._left_child = left_child
        self._right_child = right_child

class Tree(object):
    def __init__(self, min_sample=2, max_depth=tree_depth, reg_lambda=0.1):
        self._root = None
        self._min_sample = min_sample
        self._max_depth = max_depth
        self.reg_lambda = reg_lambda

    def _fit(self, X, y):
        self._root = self._build_tree(X, y)

    def predict(self, x):
        """给定输入样本，预测其类别或者输出"""
        if x.shape[0] != 1:
            sys.exit('文件 {} 第 {} 行发生错误： 请保证调用 predict 函数时传入的样本数为 1, 当前传入的样本数为：{}'
                     .format(os.path.basename(__file__), sys._getframe().f_lineno - 2, x.shape[0]))
        return self._predict(x, self._root)

    def _build_tree(self, X, y, cur_depth=0):

        # 如果树深达到限制，则返回叶节点
        if cur_depth >= tree_depth:
            node_val = self._calc_node_val(X)
            return TreeNode(node_val=node_val)

        if np.unique(y).shape[0] == 1:
            node_val = self._calc_node_val(X)
            return TreeNode(node_val=node_val)

        n_sample = X.shape[0]
        n_feature = X.shape[1] - 1


        max_gain = 0
        best_feature_idx = None
        best_feature = None
        best_feature_val = None


        feature_list = []
        for item in (x for x in X.columns if x not in ['grad']):
            feature_list.append(item)

        if feature_list is None:
            sys.exit('文件 {} 第 {} 行发生错误： feature_list为空'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))


        if n_sample >= self._min_sample:

            for i in range(n_feature):
                feature_value = np.unique(X.iloc[:, i])

                for fea_val in feature_value:

                    X_left = X[X.iloc[:, i] <= fea_val]
                    y_left = y[X.iloc[:, i] <= fea_val]


                    X_right = X[X.iloc[:, i] > fea_val]
                    y_right = y[X.iloc[:, i] > fea_val]

                    if X_left.shape[0] > 0 and y_left.shape[0] > 0 and X_right.shape[0] > 0 and y_right.shape[0] > 0:
                        before_divide = self._calc_evaluation(X.loc[:, 'grad'].sum(), X.shape[0], self.reg_lambda)
                        after_divide_left = self._calc_evaluation(X_left.loc[:, 'grad'].sum(), X_left.shape[0], self.reg_lambda)
                        after_divide_right = self._calc_evaluation(X_right.loc[:, 'grad'].sum(), X_right.shape[0], self.reg_lambda)
                        if (after_divide_left + after_divide_right - before_divide) / 2 > max_gain:
                            max_gain = (after_divide_left + after_divide_right - before_divide) / 2
                            best_feature_idx = i
                            best_feature = feature_list[i]
                            best_feature_val = fea_val


        if max_gain > 0:


            X_left = X[X[best_feature] <= best_feature_val]
            y_left = y[X[best_feature] <= best_feature_val]


            X_right = X[X[best_feature] > best_feature_val]
            y_right = y[X[best_feature] > best_feature_val]


            left_child = self._build_tree(X_left, y_left, cur_depth + 1)

            right_child = self._build_tree(X_right, y_right, cur_depth + 1)

            return TreeNode(feature_idx=best_feature_idx, feature_val=best_feature_val,
                            left_child=left_child, right_child=right_child)


        node_val = self._calc_node_val(X)
        return TreeNode(node_val=node_val)

    def _predict(self, x, tree=None):
        """给定输入预测输出"""
        # 根结点
        if tree is None:
            tree = self._root

        # 叶结点直接返回预测值
        if tree._node_val is not None:
            return tree._node_val

        # 用该结点对应的划分索引获取输入样本在对应特征上的取值
        feature_val = x.iloc[:, tree._feature_idx].values
        if feature_val <= tree._feature_val:
            return self._predict(x, tree._left_child)
        return self._predict(x, tree._right_child)

    def _calc_node_val(self, x):
        """根据当前节点样本一阶导数值、二阶导数值、正则参数计算叶节点的值"""
        gi = x.loc[:, 'grad'].sum()
        hi = x.shape[0]
        return (- gi / (hi + self.reg_lambda)) * 0.1

    def _calc_evaluation(self, gi_sum, hi_sum, lam):
        return np.power(gi_sum, 2) / (hi_sum + lam)

class xgbRegression(Tree):
    def __init__(self, n_estimator=50, max_depth=tree_depth, reg_lambda=0.1, min_sample=2):
        super().__init__()

        self._n_estimator = n_estimator # 树的数量
        self.max_depth = max_depth
        self._reg_lambda = reg_lambda # 正则参数
        self._min_sample = min_sample # 最小样本数
        self._trees = []

        for _ in range(self._n_estimator):
            tree = Tree(min_sample, max_depth)
            self._trees.append(tree)

    def fit(self, X, y):
        """模型训练"""

        # 重置行索引，便于拼接一阶导列
        X = X.reset_index(drop=True)
        # 重置列索引
        y = y.reset_index(drop=True)

        # 排除一阶导名字和某个特征名相同
        if 'grad' in X.columns:
            sys.exit('文件 {} 第 {} 行发生错误： 数据集有一个特征名为: "grad"'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

        n_samples = X.shape[0]

        # 每棵树都有删减样本
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)

        shrinkage = 0.1

        # predict 的初始化
        predict_value = np.zeros((n_samples, 1))
        # predict_value = np.ones((X.shape[0], 1)) * y.mean()[0]

        # 构造有 'grad' 的 X
        gi =  -y_copy.values
        x_gi = pd.DataFrame(gi, columns=['grad'])
        X_G = pd.concat([X_copy, x_gi], axis=1)

        y_pre = pd.DataFrame(predict_value, columns=['predict_value'])
        y_and_pre = pd.concat([y_copy, y_pre], axis=1)

        for i in range(self._n_estimator):

            # 显示进度
            # if i % 10 == 0:
            #     print('正在训练Tree {} ...'.format(i + 1))

            # # 根据之前所有树的预测值进行训练
            # self._trees[i]._fit(X, y)
            #
            # # 保存当前树对所有样本的预测值
            # # predict_value = np.zeros((X.shape[0], 1))
            # for j in range(len(predict_value)):
            #     predict_value[j] += self._trees[i].predict(X[j:j + 1])
            #
            # # 更新 gi = y_pred_tree_i - yi
            # for k in range(len(predict_value)):
            #     X.iloc[k, -1] = predict_value[k] - y.iloc[k, 0]

            te = (i + 1) % self._n_estimator
            if te == 1:
                X_G = copy.deepcopy(X_G)
                y_and_pre = copy.deepcopy(y_and_pre)

            randomly_pick = int(n_samples * shrinkage * np.power(1 - shrinkage, te - 1) / (1 - np.power(1 - shrinkage, self._n_estimator - 1)))
            if randomly_pick > X_G.shape[0]:
                choose_samples_list = list(range(X_G.shape[0]))
                X_new = X_G
                y_new = y_and_pre
            else:
                choose_samples_list = random.sample(range(X_G.shape[0]), randomly_pick)
                X_new = pd.DataFrame(X_G, index=choose_samples_list)
                y_new = pd.DataFrame(y_and_pre, index=choose_samples_list)


            X_copy.drop(choose_samples_list, inplace=True)
            X_G.drop(choose_samples_list, inplace=True)
            y_and_pre.drop(choose_samples_list, inplace=True)

            X_copy = X_copy.reset_index(drop=True)
            X_G = X_G.reset_index(drop=True)
            y_and_pre = y_and_pre.reset_index(drop=True)

            # TRAIN
            self._trees[i]._fit(X=X_new, y=y_new)


            for j in range(X_copy.shape[0]):
                y_and_pre.iloc[j, 1] += self._trees[i].predict(X_copy[j:j + 1])

            if i == 0:
                shrinkage_list = np.zeros((n_samples, 1))
                for s in range(y_and_pre.shape[0]):
                    if abs(y_and_pre.iloc[s, 1]) < abs(y_and_pre.iloc[s, 0]):
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 1]) / abs(y_and_pre.iloc[s, 0])
                    else:
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 0]) / abs(y_and_pre.iloc[s, 1])
                shrinkage = np.mean(shrinkage_list)

                if shrinkage > 1:  # shrinkage > 1 laplace 的 scale 参数就会小于0
                    meet_the_requirements = []
                    for r in shrinkage_list:
                        # if r > 0 and r < 1:
                        if r < 1:
                            meet_the_requirements.append(r)
                    shrinkage = np.mean(meet_the_requirements)

            # UPDATE gi = y_pred_tree_i - yi
            for k in range(X_G.shape[0]):
                X_G.iloc[k, -1] = y_and_pre.iloc[k, 1] - y_and_pre.iloc[k, 0]

    def predict(self, x):
        y_pred = [0] * x.shape[0]
        for tree in self._trees:
            for i in range(x.shape[0]):
                y_pred[i] += tree.predict(x[i:i+1])
        return y_pred










































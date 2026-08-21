import numpy as np
import pandas as pd
import sys
import os
import copy
import random

from self_tool import metrics_science

tree_high = 5

"""
XGBoost 中的每个基学习器：决策树
"""
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
    def __init__(self, min_sample=2, max_depth=tree_high, reg_lambda=0.1):
        """
        min_sample: 当数据集样本数少于min_sample时不再划分
        min_gain: 最小增益
        max_depth: 树的最大高度
        """
        self._root = None
        self._min_sample = min_sample
        self._max_depth = max_depth
        self.reg_lambda = reg_lambda

    def _fit(self, X, y, exp, internal_rate, shrinkage, lap1_rate, n_trees, cur_tree):
        """
        模型训练
        X: 训练集, DataFrame
        y: 训练集标签, DataFrame
        """
        self._root = self._build_tree(X, y, exp*internal_rate, exp*(1-internal_rate), shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=0)

    def predict(self, x):
        """给定输入样本，预测其类别或者输出"""
        if x.shape[0] != 1:
            sys.exit('文件 {} 第 {} 行发生错误： 请保证调用 predict 函数时传入的样本数为 1, 当前传入的样本数为：{}'
                     .format(os.path.basename(__file__), sys._getframe().f_lineno - 2, x.shape[0]))
        return self._predict(x, self._root)

    def _build_tree(self, X, y, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth):
        """
        构建树
        X: 训练数据集
        y: X 对应的标签
        exp_internal：内部节点隐私预算
        exp_leaf：叶子节点隐私预算
        lap1_rate：叶子节点laplace1噪声隐私预算比率
        n_trees: 训练总颗数
        cur_tree：当前第几颗树
        cur_depth: 当前树的高度
        """

        # 重置索引
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # 如果树深达到限制，则返回叶节点
        if cur_depth >= tree_high:
            node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
            return TreeNode(node_val=node_val)

        # 如果子树只剩下一个类别时则置为叶结点
        if np.unique(y).shape[0] == 1:
            node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
            return TreeNode(node_val=node_val)

        # 定义一个列表, 每个元素是: [(feature, cut_value): gain]
        exponent = []

        n_sample = X.shape[0]
        n_feature = X.shape[1] - 1

        # 存样本特征
        feature_list = []
        # 'grad'、'hess'保存当前样本一阶导和二阶导
        for item in (x for x in X.columns if x not in ['grad']):
            feature_list.append(item)

        if feature_list is None:
            sys.exit('文件 {} 第 {} 行发生错误： feature_list为空'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

        # 样本数至少大于给定阈值，且当前树的高度不能超过阈值才继续划分
        if n_sample >= self._min_sample:
            # 遍历每一个特征
            for i in range(n_feature):
                feature_value = np.unique(X.iloc[:, i])
                # 遍历该特征的每一个值
                for fea_val in feature_value:
                    # 左边划分
                    X_left = X[X.iloc[:, i] <= fea_val]
                    y_left = y[X.iloc[:, i] <= fea_val]

                    # 右边划分
                    X_right = X[X.iloc[:, i] > fea_val]
                    y_right = y[X.iloc[:, i] > fea_val]

                    if X_left.shape[0] > 0 and y_left.shape[0] > 0 and X_right.shape[0] > 0 and y_right.shape[0] > 0:
                        # 划分前的gain
                        before_divide = self._calc_evaluation(X.loc[:, 'grad'].sum(), X.shape[0], self.reg_lambda)
                        # 划分后的gain
                        after_divide_left = self._calc_evaluation(X_left.loc[:, 'grad'].sum(), X_left.shape[0], self.reg_lambda)
                        after_divide_right = self._calc_evaluation(X_right.loc[:, 'grad'].sum(), X_right.shape[0], self.reg_lambda)
                        if (after_divide_left + after_divide_right) > before_divide:
                            gain = (after_divide_left + after_divide_right - before_divide) / 2
                            exponent.append([(feature_list[i], fea_val), gain])

        k = []
        v = []
        for i in range(len(exponent)):
            k.append(exponent[i][0])
            v.append(exponent[i][1])

        if len(exponent) != 0 and len(v) != 0:

            # 运用指数机制输出最优分裂特征和分裂值
            optimal_index = metrics_science.exp(v, exp_internal / self._max_depth, 2)

            best_feature = k[optimal_index][0]
            best_feature_idx = feature_list.index(best_feature)
            best_feature_val = k[optimal_index][1]

            # 左边划分
            X_left = X[X[best_feature] <= best_feature_val]
            y_left = y[X[best_feature] <= best_feature_val]

            # 右边划分
            X_right = X[X[best_feature] > best_feature_val]
            y_right = y[X[best_feature] > best_feature_val]

            # 构建左子树
            left_child = self._build_tree(X_left, y_left, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=cur_depth+1)
            # 构建右子树
            right_child = self._build_tree(X_right, y_right, exp_internal, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree, cur_depth=cur_depth+1)

            return TreeNode(feature_idx=best_feature_idx, feature_val=best_feature_val,
                            left_child=left_child, right_child=right_child)

        # 样本数少于给定阈值，或者树的高度超过阈值，或者未找到合适的划分则置为叶结点
        node_val = self._calc_node_val(X, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree)
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
    #裁剪叶子结点值
    def _calc_node_val(self, x, y, exp_leaf, shrinkage, lap1_rate, n_trees, cur_tree):
        c = np.power(1 - shrinkage, cur_tree - 1)
        # new scheme 样本梯度裁剪
        for i in range(x.shape[0]):
            if abs(x.iloc[i, -1]) > c:
                if x.iloc[i, -1] > 0:
                    x.iloc[i, -1] = c
                else:
                    x.iloc[i, -1] = -c

        gi = (x.loc[:, 'grad'] + np.random.laplace(0, 2*c / exp_leaf)).sum()
        hi = x.shape[0]

        return - gi / (hi + self.reg_lambda)

    def _calc_evaluation(self, gi_sum, hi_sum, lam):
        return np.power(gi_sum, 2) / (hi_sum + lam)

class xgbRegression(Tree):
    def __init__(self, n_estimator=50, trees_in_ensemble=50, max_depth=tree_high, reg_lambda=0.1, learning_rate=0.1, min_sample=2):
        super().__init__()

        self._n_estimator = n_estimator # 树的数量
        self._trees_in_ensemble = trees_in_ensemble  # 一个集合里树的数量
        self.max_depth = max_depth
        self._reg_lambda = reg_lambda # 正则参数
        self._l_r = learning_rate # 学习率
        self._min_sample = min_sample # 最小样本数
        self._trees = []

        for _ in range(self._n_estimator):
            tree = Tree(min_sample, max_depth)
            self._trees.append(tree)

    def fit(self, X, y, exp, internal_rate, lap1_rate):
        # exp：总隐私预算
        # internal_rate：内部节点隐私预算比率
        # lap1_rate：叶子节点laplace1 噪声隐私预算比率

        # 排除一阶导名字和某个特征名相同
        if 'grad' in X.columns:
            sys.exit('文件 {} 第 {} 行发生错误： 数据集有一个特征名为: "grad"'.format(os.path.basename(__file__), sys._getframe().f_lineno - 1))

        """模型训练"""
        n_samples = X.shape[0]

        # 重置行索引，便于拼接一阶导列
        X = X.reset_index(drop=True)
        # 重置列索引
        y = y.reset_index(drop=True)

        # 每棵树都有删减样本
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)

        # 衰减率
        shrinkage = 0.1

        # 所有树都不能用的样本
        delete_index_list = None

        # predict 的初始化
        # predict_value = np.ones((X.shape[0], 1)) * y.mean()[0]
        # predict_value = [0] * n_samples
        predict_value = np.zeros((n_samples, 1))

        # gi 绑定 X_copy
        gi =  -y_copy.values
        x_gi = pd.DataFrame(gi, columns=['grad'])
        X_G = pd.concat([X_copy, x_gi], axis=1)

        # predict 绑定 y
        y_pre = pd.DataFrame(predict_value, columns=['predict_value'])
        y_and_pre = pd.concat([y_copy, y_pre], axis=1)
        # X_G_copy = copy.deepcopy(X_G)

        for i in range(self._n_estimator):
            # 每隔 10 颗树显示一下进度
            # if i % 10 == 0:
            #     print('Training Tree {} ...'.format(i + 1))

            # samples_current_tree = int(X_G.shape[0] / (self._n_estimator - i))
            # choose_samples_list = random.sample(range(X_G.shape[0]), samples_current_tree)
            # X_new = pd.DataFrame(X_G, index=choose_samples_list)
            # y_new = pd.DataFrame(y_and_pre, index=choose_samples_list)

            # 用 AAAI 取样本方式
            te = (i + 1) % self._trees_in_ensemble
            if te == 1:
                X_G = copy.deepcopy(X_G)
                y_and_pre = copy.deepcopy(y_and_pre)

            randomly_pick = int(n_samples * shrinkage * np.power(1 - shrinkage, te-1) / (1 - np.power(1 - shrinkage, self._trees_in_ensemble-1)))
            if randomly_pick > X_G.shape[0]:
                choose_samples_list = list(range(X_G.shape[0]))
                X_new = X_G
                y_new = y_and_pre
            else:
                choose_samples_list = random.sample(range(X_G.shape[0]), randomly_pick)
                X_new = pd.DataFrame(X_G, index=choose_samples_list)
                y_new = pd.DataFrame(y_and_pre, index=choose_samples_list)

            # 删除不符样本
            X_delete, y_delete, delete_index_list = metrics_science.del_samples(X_new, y_new, 1)

            # 用过的样本索引
            used_index_list = [i for i in choose_samples_list if i not in delete_index_list]

            # 删除用过的样本
            X_copy.drop(used_index_list, inplace=True)
            X_G.drop(used_index_list, inplace=True)
            y_and_pre.drop(used_index_list, inplace=True)
            # 重置索引
            X_copy = X_copy.reset_index(drop=True)
            X_G = X_G.reset_index(drop=True)
            y_and_pre = y_and_pre.reset_index(drop=True)

            # 根据之前所有树的预测值进行训练

            self._trees[i]._fit(X=X_delete,
                                y=y_delete,
                                exp=exp,
                                internal_rate=internal_rate,
                                shrinkage = shrinkage,
                                lap1_rate=lap1_rate,
                                n_trees=self._n_estimator,
                                cur_tree=i+1)

            # 保存剩余样本在之前所有树上的预测值
            for j in range(X_copy.shape[0]):
                y_and_pre.iloc[j, 1] += self._trees[i].predict(X_copy[j:j + 1])

            # 求衰减率
            if i == 0:
                shrinkage_list = np.zeros((n_samples, 1))
                for s in range(y_and_pre.shape[0]):
                    if abs(y_and_pre.iloc[s, 1]) < abs(y_and_pre.iloc[s, 0]):
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 1]) / abs(y_and_pre.iloc[s, 0])
                    else:
                        shrinkage_list[s] = abs(y_and_pre.iloc[s, 0]) / abs(y_and_pre.iloc[s, 1])
                shrinkage = np.mean(shrinkage_list)

                # if shrinkage < 0.3 or shrinkage > 0.4:
                #     shrinkage = 0.35

                if shrinkage > 1:  # shrinkage > 1 laplace 的 scale 参数就会小于0
                    meet_the_requirements = []  # 存符合条件的衰减率
                    for r in shrinkage_list:
                        # if r > 0 and r < 1:
                        if r < 1:
                            meet_the_requirements.append(r)
                    shrinkage = np.mean(meet_the_requirements)

            # 更新 gi = y_pred_tree_i - yi
            for k in range(X_G.shape[0]):
                X_G.iloc[k, -1] = y_and_pre.iloc[k, 1] - y_and_pre.iloc[k, 0]

        print('\033[0;32m 隐私预算为 {} 时训练 {} 颗树删除样本比例为：{}\033[0m'.format(exp, self._n_estimator, len(delete_index_list)/n_samples))
        return len(delete_index_list)/n_samples

    def predict(self, x):
        """给定输入样本，预测输出"""
        y_pred = [0] * x.shape[0]
        for tree in self._trees:
            for i in range(x.shape[0]):
                y_pred[i] += tree.predict(x[i:i+1])
        return y_pred












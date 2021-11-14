#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: myfile.py
@time: 2021-11-13 11:36
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from loguru import logger


class WeakClassifier():
    def __init__(self):
        self.feature_idx = None  # 哪个划分特征
        self.feature_val = None  # 划分的值
        self.threshold_type = None  # 划分的方式
        self.y_pred = None  # 分类器分类结果
        self.error_rate = None  # 该弱分类器数据集上的错误率
        self.alpha = None  # 弱分类器的权重


class AdaboostScratch():
    def __init__(self, T):
        self.T = T  # 训练轮数
        self.classifier = list()

    def fit(self, X, y):
        numSample, numFeature = X.shape
        D = np.ones(numSample) / numSample

        for _ in range(self.T):
            # 基于现在的权重使用弱分类器进行训练
            wc = WeakClassifier()  # 构造一个空的弱分类器
            minError = np.inf  # 记录最小的误差
            # 遍历每一维的特征
            for i in range(numFeature):
                feature_val = np.unique(X[:, i])
                for fea_val in feature_val:
                    for threshold_type in ['lt', 'gt']:
                        y_pred = self.stumpClassify(X, i, fea_val, threshold_type)
                        # 计算错误率
                        errorSample = np.ones(numSample)
                        y_pred = y_pred.reshape(160,)
                        errorSample[y_pred == y] = 0
                        weightError = np.dot(D, errorSample)
                        if weightError < minError:
                            minError = weightError
                            wc.feature_idx = i
                            wc.feature_val = fea_val
                            wc.threshold_type = threshold_type
                            wc.y_pred = y_pred
                            wc.error_rate = weightError

            if wc.error_rate > 0.5:
                continue
            # 更新学习器权重
            wc.alpha = 0.5 * np.log((1 - wc.error_rate) / max(wc.error_rate, 1e-16))  # 防止分母为零
            # 更新样本分布
            D *= np.exp(-wc.alpha * y * wc.y_pred)
            D /= np.sum(D)

            self.classifier.append(wc)

    def predict(self, x):
        y_pred = 0
        for cls in self.classifier:
            pred = 1
            if cls.threshold_type == 'lt':
                if x[cls.feature_idx] <= cls.feature_val:
                    pred = -1
            else:
                if x[cls.feature_idx] > cls.feature_val:
                    pred = -1
            y_pred += cls.alpha * pred

        return np.sign(y_pred)

    def stumpClassify(self, X, feature_idx, fea_val, threshold_type):
        # 决策树桩的分类函数
        y_pred = np.ones((X.shape[0], 1))
        if threshold_type == 'lt':
            y_pred[X[:, feature_idx] <= fea_val] = -1
        else:
            y_pred[X[:, feature_idx] > fea_val] = -1
        return y_pred

def loadDataSet(filename):
    """
    读取数据，并输出特征集和标签
    :param filename:数据路径
    :return: data, label
    """
    dataset = np.loadtxt(filename)
    linenum = dataset.shape[1]
    data = dataset[:, 0:linenum - 1]
    label = dataset[:, -1]
    return data, label


def main():
    parser = argparse.ArgumentParser(description='adaboost算法实现参数')
    parser.add_argument('--T',type=int, default=30, help='训练多少个弱分类器')
    args = parser.parse_args()

    X, y = loadDataSet('./data/heart.txt')
    y[y == 2] = -1

    x_train, x_test, y_train, y_test = train_test_split(X[:200],y[:200], train_size=0.8,shuffle=True)

    model = AdaboostScratch(args.T)
    model.fit(x_train,y_train)

    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(x_test[i])
        if y_pred == y_test[i]:
            n_right += 1
        else:
            logger.info("样本真实标签为：{}，但是模型的预测标签为：{}".format(y_test[i],y_pred))
    logger.info("Adaboost算法在测试集上的准确率为：{}%".format(n_right * 100 / n_test))

    sk_model = AdaBoostClassifier(n_estimators=args.T)
    sk_model.fit(x_train,y_train)
    logger.info("sklearn模型在测试集上的准确率为：{}%".format(100 * sk_model.score(x_test,y_test)))


if __name__ == "__main__":
    main()




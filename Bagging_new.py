#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: myfile.py
@time: 2021-11-13 11:36
"""
import numpy as np
from numpy import random
import pandas as pd
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class BaggingScratch():
    def __init__(self, T, clf):
        self.T = T
        self.clf = clf
        self.classifiers = []

    def bootStrapSampling(self, X, y):
        """
        有放回的采样
        :return:
        """
        numSample, numFeature = X.shape
        samples = np.zeros((numSample, numFeature))
        labels = np.zeros((numSample, 1))
        for i in range(numSample):
            randomNum = random.randint(0, numSample)
            samples[i] = (X[randomNum])
            labels[i] = (y[randomNum])
        return samples, labels

    def fit(self, X, y):
        for i in range(self.T):
            samples, labels = self.bootStrapSampling(X, y)
            self.clf.fit(samples, labels)
            self.classifiers.append(self.clf)
        return self

    def predict(self, x):
        result = 0
        for i, clf in enumerate(self.classifiers):
            x_transfer = x.reshape((1, -1))
            pred = clf.predict(x_transfer)
            result += pred
        return np.sign(np.sum(result))


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
    parser = argparse.ArgumentParser(description='bagging算法实现参数')
    parser.add_argument('--T', type=int, default=30, help='训练多少轮')
    args = parser.parse_args()

    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 8,
        'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }

    X, y = loadDataSet('./data/heart.txt')
    y[y == 2] = -1

    x_train, x_test, y_train, y_test = train_test_split(X[:200], y[:200], train_size=0.8, shuffle=True)

    baseModel = RandomForestClassifier(**rf_params)

    model = BaggingScratch(args.T, baseModel)
    model.fit(x_train, y_train)

    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(x_test[i])
        if y_pred == y_test[i]:
            n_right += 1
        else:
            logger.info("样本真实标签为：{}，但是模型的预测标签为：{}".format(y_test[i], y_pred))
    logger.info("Bagging算法在测试集上的准确率为：{}%".format(n_right * 100 / n_test))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xinquan Chen
@contact:XinquanChen0117@163.com
@file: myfile.py
@time: 2021-11-13 11:36
"""
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import clone
from loguru import logger

# 五个初级分类器
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

# 使用决策树作为我们的次级分类器
from sklearn.tree import DecisionTreeClassifier

class StackingScratch():
    def __init__(self, n, priClfs, secondClf):
        self.n = n
        self.priClfs = priClfs
        self.secondClf = secondClf

    def getStacking(self, priClf, priClfIndex, x_train, y_train, x_test):
        """
        这个函数主要是产生次级的训练集和测试集
        :return:
        """
        kf = KFold(n_splits=self.n)
        train_num, test_num = x_train.shape[0], x_test.shape[0]
        secondaryTrain = np.zeros((train_num,))  # A
        secondaryTest = np.zeros((test_num,))  # B
        testSet = np.zeros((test_num, self.n))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_tst, y_tst = x_train[test_index], y_train[test_index]

            priClf.fit(x_tra, y_tra)

            secondaryTrain[test_index] = priClf.predict(x_tst)
            testSet[:, i] = priClf.predict(x_test)

        secondaryTest[:] = testSet.mean(axis=1)
        self.priClfs[priClfIndex] = priClf

        return secondaryTrain, secondaryTest

    def fit(self, x_train, y_train, x_test):

        numTrain = x_train.shape[0]
        numTest = x_test.shape[0]
        numClfs = len(self.priClfs)
        secondaryTrainSet = np.zeros((numTrain, numClfs))
        secondaryTestSet = np.zeros((numTest, numClfs))
        for i, priClf in enumerate(self.priClfs):
            priClf = clone(priClf)
            secondaryTrain, secondaryTest = self.getStacking(priClf, i, x_train, y_train, x_test)
            secondaryTrainSet[:, i] = secondaryTrain
            secondaryTestSet[:, i] = secondaryTest
            # self.priClfs[i] = clone(model)

        self.secondClf.fit(secondaryTrainSet, y_train)
        # self.secondClf = clone(self.secondClf)

        return self

    def predict(self, x):
        numClfs = len(self.priClfs)
        priFeatures = np.zeros((numClfs,))
        for i, model in enumerate(self.priClfs):
            #x.reshape((1, -1))
            priFeatures[i] = model.predict(x)
        priFeatures = priFeatures.reshape((1, -1))

        return self.secondClf.predict(priFeatures)

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
    parser = argparse.ArgumentParser(description='stacking算法实现参数')
    parser.add_argument('--n',type=int, default=5, help='n折交叉验证')
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

    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
    }

    rf = RandomForestClassifier(**rf_params)
    et = ExtraTreesClassifier(**et_params)
    ada = AdaBoostClassifier(**ada_params)
    gb = GradientBoostingClassifier(**gb_params)
    svc = SVC(**svc_params)

    priClfs = [rf, et, ada, gb, svc]

    secondClf = DecisionTreeClassifier()

    model = StackingScratch(args.n, priClfs, secondClf)

    X, y = loadDataSet('./data/heart.txt')
    y[y == 2] = -1

    x_train, x_test, y_train, y_test = train_test_split(X[:200],y[:200], train_size=0.8,shuffle=True)

    model.fit(x_train,y_train,x_test)

    n_test = x_test.shape[0]
    n_right = 0
    for i in range(n_test):
        x_test_transfer = x_test[i].reshape((1,-1))
        y_pred = model.predict(x_test_transfer)
        if y_pred == y_test[i]:
            n_right += 1
        else:
            logger.info("样本真实标签为：{}，但是模型的预测标签为：{}".format(y_test[i],y_pred))
    logger.info("stacking算法在测试集上的准确率为：{}%".format(n_right * 100 / n_test))

if __name__ == "__main__":
    main()
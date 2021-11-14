#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017-08-28

@author: panda_zjd
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import IsolationForest

class Bagging(object):

    def __init__(self,n_estimators,estimator,rate=1.0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.rate = rate

    def Voting(self,data):          #投票法
        term = np.transpose(data)   #转置
        result =list()              #存储结果

        def Vote(df):               #对每一行做投票
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store,key=store.get)

        result= map(Vote,term)      #获取结果
        return result

    #随机欠采样函数
    def UnderSampling(self,data):
        #np.random.seed(np.random.randint(0,1000))
        data=np.array(data)
        np.random.shuffle(data)    #打乱data
        newdata = data[0:int(data.shape[0]*self.rate),:]   #切片，取总数*rata的个数，删去（1-rate）%的样本
        return newdata

    def TrainPredict(self,train,test):          #训练基础模型，并返回模型预测结果
        clf = self.estimator.fit(train[:,0:-1],train[:,-1])
        result = clf.predict(test[:,0:-1])
        return result

    #简单有放回采样
    def RepetitionRandomSampling(self,data,number):     #有放回采样，number为抽样的个数
        sample=[]
        for i in range(int(self.rate*number)):
            sample.append(data[random.randint(0,len(data)-1)])
        return sample

    def Metrics(self,predict_data,test):        #评价函数
        score = predict_data
        recall=recall_score(test[:,-1], score, average=None)    #召回率
        precision=precision_score(test[:,-1], score, average=None)  #查准率
        return recall,precision


    def MutModel_clf(self,train,test,sample_type = "RepetitionRandomSampling"):
        print("self.Bagging Mul_basemodel")
        result = list()
        num_estimators =len(self.estimator)   #使用基础模型的数量

        if sample_type == "RepetitionRandomSampling":
            print("选择的采样方法：",sample_type)
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print("选择的采样方法：",sample_type)
            sample_function = self.UnderSampling
            print("采样率",self.rate)
        elif sample_type == "IF_SubSample":
            print("选择的采样方法：",sample_type)
            sample_function = self.IF_SubSample
            print("采样率",(1.0-self.rate))

        for estimator in self.estimator:
            print(estimator)
            for i in range(int(self.n_estimators/num_estimators)):
                sample=np.array(sample_function(train,len(train)))       #构建数据集
                clf = estimator.fit(sample[:,0:-1],sample[:,-1])
                result.append(clf.predict(test[:,0:-1]))      #训练模型 返回每个模型的输出

        score = self.Voting(result)
        recall,precosoion = self.Metrics(score,test)
        return recall,precosoion


import numpy as np
import pandas as pd
from loguru import logger


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


def buildStump(dataArr, labelArr, D):
    """
    决策树函数: 得到数据集上最佳单层决策树的模型
    :param dataArr: 特征集合
    :param labelArr: 标签集合
    :param D: 样本分布
    :return:
        bestStump:最优的分类器模型
        minError:错误率
        bestClassEst:训练后的结果集合
    """
    # 转换数据
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    # m行，n列
    m, n = np.shape(dataMat)
    # 初始化数据
    numSteps = 10.0 # 步长
    bestStump = {} # 最佳单层决策树
    bestClassEst = np.mat(np.zeros((m, 1))) # 最佳分类结果
    minError = np.inf # 最小误差初始化为正无穷大
    # 第一层循环，循环数据集的每一个特征
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        # print('rangeMin=' + str(rangeMin) + ', rangeMax=' + str(rangeMin))
        # 计算每一份元素的个数
        stepSize = (rangeMax - rangeMin) / numSteps

        # 第二层循环，循环数据集每一个步长
        for j in range(-1, int(numSteps) + 1):
            # 对每一个不等号， 大于小于均遍历，lt:less than gt:great than
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize) # 计算阈值
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                """
                dim            表示特征列
                threshVal      表示树的分界值
                inequal        表示计算树左右颠倒的错误率的情况
                weightedError  表示整体结果的错误率
                bestClassEst   预测的最优结果
                """
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEst


def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    """
    分类函数: 将数据集，按照特征列的value进行二分法切分比较来赋值分类
    :param dataMat: Matrix数据集
    :param dimen: 特征列
    :param threshVal: 特征列要比较的值
    :param threshIneq: 分类规则
    :return: retArray 结果集
    """
    retArray = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:,dimen] > threshVal] = -1.0

    return retArray


def AdaBoostTrainDS(dataArr, labelArr, numIt=40):
    """
    AdaBoost算法实现
    :param dataArr: 特征标签集合
    :param labelArr: 分类标签集合
    :param numIt: 弱分类器个数
    :return: weakClassArr: 弱分类器的集合， aggClassEst: 预测的分类结果值
    """
    weakClassArr = []
    m = np.shape(dataArr)[0] #行个数（样本个数）
    # 初始化分布
    W = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        # 得到决策树模型（弱分类器）
        bestStump, error, classEst = buildStump(dataArr, labelArr, W)

        # alpha是计算每个分类器的权重 下面是计算和更新alpha
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        # 将stump参数存在Array中
        weakClassArr.append(bestStump)

        # 下面更新W分布
        expon = np.multiply(-1*alpha*np.mat(labelArr).T, classEst)
        W = np.multiply(W, np.exp(expon))
        W = W/W.sum()

        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        aggClassEst += alpha*classEst #当前分类结果
        aggErrors = np.multiply(np.sign(aggClassEst)!=np.mat(labelArr).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        if errorRate == 0.0:
            break

    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    分类准确率的计算
    :param datToClass:
    :param classifierArr:
    :return: np.sign(aggClassEst)
    """
    dataMat = np.mat(datToClass)
    m = np.shape(dataMat)[0]
    aggClassEst = np.mat(np.zeros((m,1)))

    # 循环多个分类器
    for i in range(len(classifierArr)):
        # 前提： 我们已经知道最佳分类器的组合
        # 通过分类器来核算每一次的分类结果，然后通过alpha*每一次的结果，得到最后的权重加和的值
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return np.sign(aggClassEst)



if __name__ == "__main__":
    # dataArr, labelArr = loadDataSet('./data/zhengqi_train.txt')
    # weakClassArr, aggClassEst = AdaBoostTrainDS(dataArr, labelArr, 40)

    # datMat = np.matrix(
    #     [[1. , 2.1],
    #      [1.5, 1.6],
    #      [1.3, 1. ],
    #      [1. , 1. ],
    #      [2. , 1. ]])
    # classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # D = np.mat(np.ones((5,1))/5)
    # bestStump, minError, bestClassEst = buildStump(datMat,classLabels,D)
    # print('bestStump:', bestStump)
    # print('minError:', minError)
    # print('bestClassEst:', bestClassEst)
    # weakClassArr, aggClassEst = AdaBoostTrainDS(datMat,classLabels,5)
    # print(weakClassArr)
    # print(aggClassEst)
    dataArr, labelArr = loadDataSet('./data/heart.txt')
    for i in range(len(labelArr)):
        if labelArr[i] == 2:
            labelArr[i] = -1
    weakClassArr, aggClassEst = AdaBoostTrainDS(dataArr, labelArr, 40)
    aggClassEst = np.sign(aggClassEst)
    for j in range(len(aggClassEst)):
        errorcount = 0
        if aggClassEst[j] == labelArr[j]:
            errorcount += 1

    logger.info("adaboost算法的错误率为：{}%".format(errorcount * 100 / len(aggClassEst)))


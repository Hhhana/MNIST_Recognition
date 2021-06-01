# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:15:54 2021

@author: Hana Luo
"""

import numpy as np
from numpy import *
import time
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import struct
import math

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

def loadDataSet():
    """
        加载并处理数据
    """
    # 从网站读取数据
    X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)  # Load data from https://www.openml.org/d/554 
    print("数据加载结束...", X.shape,y.shape) # (70000, 784) (70000,)
    #print("标签类型: ", type(y))
    # 数据处理
    random_state = check_random_state(0)  # 随机数种子
    permutation = random_state.permutation(X.shape[0]) # 随机排列数组，打乱样本
    X = X[permutation]
    y = y[permutation]
    
    X = X.reshape((X.shape[0],-1))
    # print(X.shape) # (70000, 784)
    
    train_samples = 50000 # 自定义训练样本的个数，数据量少提高训练速度
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size = train_samples, test_size = 10000)  # 拆分验证集 1000个
    # print(train_x.shape) # (5000, 784)
    # print(train_x)
    
    # 数据归一化
    '''
        fit_transform方法是fit和transform的结合，fit_transform(X_train) 意思是找出X_train的和，并应用在X_train上。
        这时对于X_test，我们就可以直接使用transform方法。因为此时StandardScaler已经保存了X_train的和。
    '''
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x, train_y, test_y


def softmax(z):
    """
        softmax函数,用于多元分类,数据集均为np.array格式
        z: m*k,m为样本数,k为类别数
    """
    [m,k] = z.shape
    p = np.zeros([m,k])
    for i in range(m):
        p[i,:] = np.exp(z[i,:]) / np.sum(np.exp(z[i,:]))  # z[i: ]第i行
    return p


def cost(theta, x, y): #x(m行n列），y（m行k列），theta（k行n列）
    """
        损失函数J_theta
        theta: 参数集k*(n+1),k为标签的类别数，n为特征数
        x: 数据集m*n
        y: 标签集m*k
        return:  损失函数结果
    """
    [k, m] = y.shape
    theta = np.matrix(theta)  #
    p = softmax(np.dot(x, theta.T))  # [m,k],P(i)为[1,k]
    p = p.T.reshape([k*m,1])
    y = y.reshape([k*m,1])
    temp_p = np.mat(np.log(p))  # 取对数
    #punish = np.sum(np.power(theta,2))  # 惩罚项，在梯度下降时求导，这里不必加上去
    cost = -1/m * np.dot(y.T, temp_p) #+ punish
    return cost  # 输出m行k列，代表m个样本，k个类别各自概率


def gradientDescent(x, y, theta, iters, alpha, regulization_rate):
    """
        参数优化过程：梯度下降法
        x: 输入
        y: 标签
        theta: 权重系数
        iters: 迭代次数
        alpha: 学习率
        regulization_rate: 正则项系数
        
    """
    COST = np.zeros((iters,1))  # 存放每次迭代后，cost值的变化
    #thetaNums = int(theta.shape[0])  # 维数，即j的取值个数n
    m = x.shape[0]
    #print(thetaNums)
    for i in range(iters):
        #bb = x*theta.T
        p = softmax(np.dot(x, theta.T));
        grad = (-1/m * np.dot(x.T, (y.T - p))).T  # [3,784],注意负号
        # 更新theta
        theta = theta - alpha*grad  #- regulization_rate * theta，正则项可以省略
        COST[i] = cost(theta, x, y)
        #每训练一次，输出当前训练步数与损失值
        '''
        print("训练次数： ", i+1)
        print("cost:", COST[i])
        print("\n")
        '''
    #返回迭代后的theta值，和每次迭代的代价函数值
    return theta, COST


def train_model(train_x, train_y, theta, learning_rate, iterationNum, numClass):
    """
    训练模型
    train_x: 训练集的输入向量m*n
    train_y: 训练集的标签向量m*1 
    theta: theta是numClass*(n+1)的向量，优化的参数向量
    learning_rate: 随机梯度下降法中的学习率lr
    iterationNum: 迭代次数
    numClass: 类别数，MNIST中为10
    # 若需要正则化可以添加参数
    """
    m = train_x.shape[0]  # train_x的行数
    #n = train_x.shape[1]  # train_x的列数
    train_x = np.insert(train_x, 0, values = 1, axis = 1)  # 扩充X向量，在最后一列后增加1向量
    real_y = np.zeros((m, numClass))  # y为m*k    
    
    # mnist每个样本的标签是一个数字，以独热码的形式将其扩充为10维的向量
    for i in np.arange(0, m):
        label = int(train_y[i])
        real_y[i, label] = 1;
    real_y = real_y.T  # k*m
    print("real_y.shape: ", real_y.shape)
    J_theta = np.zeros((iterationNum, numClass))

    # 优化参数，使得损失函数取到最小值
    theta, J_theta = gradientDescent(train_x, real_y, theta, iterationNum, learning_rate, 0) 
    return theta  # 返回的theta是n*numClass矩阵(785, 10)


'''
    二分类预测函数
    test_x: 验证集输入向量
    test_y: 验证集标签
    theta: 训练后得到的参数向量，是n*numClass的矩阵
    numClass: 分类数
    
'''
def predict(test_x, test_y, theta, numClass):
    errorCount = 0  #  预测错误的个数，用来计算误差
    test_x = np.insert(test_x, 0, values = 1, axis = 1)  # 向右扩充向量
    m = test_x.shape[0]  # 行数

    # 计算概率
    p = softmax(np.dot(test_x, theta.T))  # h_theta是m*numClass的矩阵，因为test_x是m*n，theta是n*numClass
    print("p: ",p)
    #p_max = p.max(axis = 1)  # 获得每行的最大值,h_theta_max是m*1的矩阵，列向量
    p_max_postion = p.argmax(axis = 1)#获得每行的最大值的label，和索引正好是对应的
    for i in range(m):  # 遍历每一行
        #print("label:", test_y[i])
        #print("predict: ", h_theta_max_postion[i])
        if test_y[i] != str(p_max_postion[i]):
            errorCount += 1
    
    error_rate = float(errorCount) / m
    print("error_rate", error_rate)
    print("accuracy: %.2f" % (100 * (1 - error_rate)))
    return error_rate




def mulitPredict(test_x, test_y, theta, iteration):
    """
    多次预测，打印平均误差
    test_x: 验证集输入向量
    test_y: 验证集标签
    theta: 训练后得到的参数向量，是n*numClass的矩阵
    iteration: 预测次数
    """
    numPredict = 10  # 预测次数
    errorSum = 0
    for k in range(numPredict):
        errorSum += predict(test_x, test_y, theta, iteration)
    print("after predict %d iterations the average accuracy rate is:%.2f" % (numPredict, 100 - 100*errorSum / float(numPredict)))  # 预测numPredict次的平均误差



if __name__ == '__main__':
    print("Start reading data...")
    time1 = time.time()
    # 加载数据并处理
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()
    print("read data cost",time2-time1,"second")
    
    # 多分类问题
    numClass = 10
    iteration = 1000  # BGD迭代次数
    learning_rate = 0.25 # 0.001
    n = test_x.shape[1] + 1  # 扩充向量后的列数
    regulization_rate = 0.1  # 惩罚系数
    
    # 权重向量：随机构造numClass*n的矩阵,注意n为列数，可理解为特征维数
    theta = np.zeros((numClass, n))  # theta  = np.random.rand(n,1)

    print("Start training data...")
    theta_new = train_model(train_x, train_y, theta, learning_rate, iteration, numClass)
    time3 = time.time()
    print("train data cost", time3 - time2, "second")

    print("Start predicting data...")
    '''
    predict(test_x, test_y, theta_new, iteration)
    '''
    iteration = 10
    mulitPredict(test_x, test_y, theta_new, iteration)
    time4 = time.time()
    print("predict data cost",time4 - time3,"second")
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:36:14 2021

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

'''
    加载数据并归一化
'''
def loadDataSet():
    # 从网站读取数据
    X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)  # Load data from https://www.openml.org/d/554 
    print("数据加载结束...", X.shape,y.shape) # (70000, 784) (70000,)
    print("标签类型: ", type(y))
    # 数据处理
    random_state = check_random_state(0)  # 随机数种子
    permutation = random_state.permutation(X.shape[0]) # 随机排列数组，打乱样本
    X = X[permutation]
    y = y[permutation]
    
    X = X.reshape((X.shape[0],-1))
    # print(X.shape) # (70000, 784)
    
    train_samples = 60000 # 自定义训练样本的个数，数据量少提高训练速度
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size = train_samples, test_size = 10000)  # 拆分验证集 1000个
    # print(train_x.shape) # (5000, 784)

    # 数据归一化
    '''
        fit_transform方法是fit和transform的结合，fit_transform(X_train) 意思是找出X_train的和，并应用在X_train上。
        这时对于X_test，我们就可以直接使用transform方法。因为此时StandardScaler已经保存了X_train的和。
    '''
    scaler = StandardScaler()  # zero-mean normalization
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x, train_y, test_y

'''
    sigmoid优化函数
    inX：输入向量
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


'''
    sigmoid优化
    RuntimeWarning: overflow encountered in exp
    对sigmoid函数的优化，避免出现极大的数据溢出
    
'''
def sigmoid_opt(x):
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


'''
    二分类器
'''
def classifyVector(inX, weights):  # 这里的inX相当于test_data即验证集,以回归系数和特征向量作为输入来计算对应的sigmoid
    prob = sigmoid(sum(inX*weights))  # p = sigmoid(sum(xw))
    if prob > 0.5:  # 阈值为0.5，大于则为正例
        return 1.0
    else:  # 否则为负例
        return 0.0
    
    
'''
    训练模型
    train_x: 训练集的输入向量
    train_y: 训练集的标签向量 
    theta: theta是(n+1) * numClass的向量，优化的参数向量
    learning_rate: 随机梯度下降法中的学习率lr
    iterationNum: 迭代次数
    numClass: 类别数，MNIST中为10
'''
def train_model(train_x, train_y, theta, learning_rate, iterationNum, numClass):
    m = train_x.shape[0]  # train_x的行数
    #n = train_x.shape[1]  # train_x的列数
    train_x = np.insert(train_x, 0, values = 1, axis = 1)  # 扩充X向量，在最后一列后增加1向量
    J_theta = np.zeros((iterationNum, numClass))


    for k in range(numClass):  # 第k个分类器，k = 0 ~ 9
        real_y = np.zeros((m,1))  # 训练集的label
        # index = train_y == str(k)  # index中存放的是train_y中等于k的索引
        index = np.where(train_y == str(k))
        real_y[index] = 1  # 在real_y中修改相应的index对应的值为1，先分类0和非0，然后是1和非1，……
        #print("real_y:", real_y) # 打印二分类标签
        
        # 优化参数，使得损失函数取到最小值
        for j in range(iterationNum):  # 迭代次数，即对第k个分类器的训练次数
            temp_theta = theta[:,k].reshape((785,1))  # 初始化为列向量，原值785
            # expit即sigmoid函数,内部没有处理数据过大带来的溢出问题
            # h_theta = expit(np.dot(train_x, temp_theta)).reshape((m,1))  # 60000为训练样本个数,初始值60000
            
            h_theta = sigmoid_opt(np.dot(train_x, temp_theta))  # 这是个向S
            #print("h_theta: ", h_theta)
            
            # untimeWarning: divide by zero encountered in log
            # log的输入过小会溢出，固定一下精度
            J_theta[j,k] = (np.dot(np.log(1e-5 + h_theta).T, real_y) + np.dot((1 - real_y).T, np.log(1e-5 + 1 - h_theta))) / (-m)  # 损失函数
            
            # 临时的theta：(785,1)
            temp_theta = temp_theta + learning_rate * np.dot(train_x.T, (real_y - h_theta))  # 末尾的矩阵乘法即J对theta的求梯度的向量形式
            
            theta[:, k] = temp_theta.reshape((785,))  # 保存到矩阵第k列

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
    print("test_x.shape[1] = %d" %test_x.shape[1])

    # 计算概率
    h_theta = expit(np.dot(test_x, theta))  # h_theta是m*numClass的矩阵，因为test_x是m*n，theta是n*numClass
    print("h_theta: ",h_theta)
    #h_theta_max = h_theta.max(axis = 1)  # 获得每行的最大值,h_theta_max是m*1的矩阵，列向量
    h_theta_max_postion = h_theta.argmax(axis = 1)  # 获得每行的最大值的label，和索引正好是对应的
    for i in range(m):  # 遍历每一行
        if test_y[i] != str(h_theta_max_postion[i]):
            errorCount += 1
    
    error_rate = float(errorCount) / m
    print("error_rate", error_rate)
    print("accuracy: %f" % (100 * (1 - error_rate)))
    return error_rate


'''
    多次预测，打印平均误差
    test_x: 验证集输入向量
    test_y: 验证集标签
    theta: 训练后得到的参数向量，是n*numClass的矩阵
    iteration: 这里为类别数
'''
def mulitPredict(test_x, test_y, theta, iteration):
    numPredict = 10  # 预测次数
    errorSum = 0
    for k in range(numPredict):
        errorSum += predict(test_x, test_y, theta, iteration)
    print("after predict %d iterations the average accuracy rate is:%f" % (numPredict, 100 - 100 * errorSum / float(numPredict)))  # 预测numPredict次的平均误差
    
    
if __name__ == '__main__':
    print("Start reading data...")
    time1 = time.time()
    # 加载数据并处理
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()
    print("read data cost",time2-time1,"second")
    
    # 多分类问题
    numClass = 10
    iteration = 1000  # BGD的迭代次数
    learning_rate = 0.001 # 0.001
    n = test_x.shape[1] + 1  # 扩充向量后的列数
    
    # 权重向量：随机构造n*numClass的矩阵,因为有numClass个分类器，所以应该返回的是numClass个列向量（n*1），注意n为列数，可理解为特征维数
    theta = np.zeros((n,numClass))  # theta  = np.random.rand(n,1)

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
"""
Logistic Regression:
loss function: logistic loss, 对率损失
wTxi + b = ln(y/(1-y)), 线性模型直接输出的是对率
训练，最大化“对数似然”，构建似然函数，使用梯度下降 或者 牛顿法 优化
"""

import numpy as np
from sklearn.datasets import load_iris
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import linear_model

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def cost(h, y):
    return (-y * np.log(h) - (1-y) * np.log(1-h)).mean()

def gradient(X,h,y):
    return np.dot(X.T, (h-y))/y.shape[0]

def logistic_reg(X,y,theta,alpha,iters):
    cost_array = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(np.dot(X,theta).astype(float))
        cost_num = cost(h,y)
        cost_array[i] = cost_num
        grad = gradient(X,h,y)
        theta = theta - grad*alpha
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

def run():
    # data = load_iris()
    # print(data)
    data = pd.read_csv('../irisdata.csv')
    print(data)
    X = data[['length','width']]
    y = data['Type']
    print(X.shape)
    print(y.shape)
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    print(X)
    print(y)
    theta = np.zeros(X.shape[1])
    lr = 0.01
    iters = 10000
    h = sigmoid(np.dot(X,theta).astype(float))
    print(f"starting cost: {cost(h,y)}")
    theta, costs = logistic_reg(X,y,theta,lr,iters)
    plotChart(iters, costs)
    h = sigmoid(np.dot(X,theta).astype(float))
    print("final cost: ", cost(h,y))


if __name__ == '__main__':
    run()
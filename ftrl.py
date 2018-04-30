# -*- coding: utf-8 -*-
# Toy-Machine-Learning
#
# @author: Shangru 
# @email: draco.mystack@gmail.com
# @date: 2018/04/29 21:00

import pandas as pd
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class FTRL(object):

    def __init__(self, alpha, beta, lambda_1, lambda_2, d):
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.d = d
        self.z = np.zeros(d)
        self.n = np.zeros(d)
        self.w = np.zeros(d)
        self.max_iter = 5000


    def get_nonzero_idx(self, x):
        index = []
        for i, xi in enumerate(x):
            if xi != 0:
                index.append(i)
        return index

    def update(self, x, y):
        max_len = len(x)
        for t in range(self.max_iter):
            x_idx = np.random.randint(0, max_len)
            xt = x.iloc[x_idx, :]
            yt = y.iloc[x_idx]
            g = np.zeros(len(xt))
            sigma = np.zeros(len(xt))

            I = self.get_nonzero_idx(xt)
            for i in I:
                if np.abs(self.z[i]) <= self.lambda_1:
                    self.w[i] = 0
                else:
                    self.w[i] = -1.0/((self.beta + np.sqrt(self.n[i]))/self.alpha + self.lambda_2)\
                        *(self.z[i] - np.sign(self.z[i])*self.lambda_1)
            pt = sigmoid(np.dot(xt, self.w))
            for i in I:
                g[i] = (pt - yt)*xt[i]
                sigma[i] = 1.0/self.alpha*(np.sqrt(self.n[i] + np.power(g[i], 2)) - np.sqrt(self.n[i]))
                self.z[i] = self.z[i] + g[i] - sigma[i]*self.w[i]
                self.n[i] = self.n[i] + np.power(g[i], 2)
        return

def accuracy(pred, real):
    acc = 0
    size = len(pred)
    for i in range(size):
        pred[i] = 1 if pred[i] > 0.5 else 0
        if pred[i] == int(real.iloc[i]):
            acc += 1
    return 1.0*acc / size


if __name__ == '__main__':
    data = pd.read_csv("./dataset/train.txt", delimiter=' ', names=['x1', 'x2', 'x3', 'x4', 'y'])
    X = data[['x1', 'x2', 'x3', 'x4']]
    y = data[['y']]

    ftrl = FTRL(alpha=0.1, beta=1.0, lambda_1=1.0, lambda_2=1.0, d=4)
    ftrl.update(X, y)
    w = ftrl.w
    print(w)
    pred = sigmoid(np.matmul(X, w))
    print(accuracy(pred, y))

    ftrl = FTRL(alpha=0.5, beta=1, lambda_1=0, lambda_2=0, d=4)
    ftrl.update(X, y)
    w = ftrl.w
    print(w)
    pred = sigmoid(np.matmul(X, w))
    print(accuracy(pred, y))


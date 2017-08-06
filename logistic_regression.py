# coding: utf-8
"""
    Logistic Regression with L2-regularization

    @author: Shangru
"""
import numpy as np
from sklearn import datasets

def load_data():
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    r = np.random.permutation(len(y))
    X = X[r, :]
    y = y[r]

    train_X = X[:int(len(y)*0.7)]
    train_y = y[:int(len(y)*0.7)]
    test_X = X[int(len(y)*0.7):]
    test_y = y[int(len(y)*0.7):]

    print train_X.shape, test_X.shape
    
    return train_X, train_y, test_X, test_y

class LogisticRegression(object):
    def __init__(self, penalty='l2', max_iter=100, optimizer='sgd'):
        self.w = None
        self.alpha = 0.01
        self.penalty = penalty
        self.max_iter = max_iter
        self.optimizer = optimizer

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.random.randn(n)
        for epoch in xrange(self.max_iter):
            tol = loss(self.sigmoid(np.dot(self.w, X.T)), y)
            print "Epoch: {}, loss: {}".format(epoch, tol)
            if tol < 1e-5:
                print "early stop!"
                break
            if self.optimizer == "sgd":
                for i in xrange(m):
                    pred = self.sigmoid(np.dot(self.w.T, X[i]))
                    grad = (pred - y[i])*X[i]
                    self.w = self.w - self.alpha*grad - 1/m*self.w
            elif self.optimizer == "bgd":
                pred = self.sigmoid(np.dot(self.w, X.T)) # 1*n n*m = 1*m
                grad = np.dot(X.T, pred - train_y) # n*m  1*m
                self.w = self.w - self.alpha*grad - 1/m*self.w
        return
    
    def predict(self, X):
        m = (X.shape)[0]
        pred = np.zeros(m)
        for i in xrange(m):
            pred[i] = 1 if self.sigmoid(np.dot(self.w.T, X[i])) >= 0.5 else 0
        return pred

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

def loss(pred, real):
    if len(pred) != len(real):
        return
    error = 0
    for i in xrange(len(pred)):
        error += abs(pred[i] - real[i])
    return error/len(real)

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data()
    lr = LogisticRegression(optimizer="sgd")
    lr.fit(train_X, train_y)
    pred = lr.predict(test_X)
    print loss(pred, test_y)
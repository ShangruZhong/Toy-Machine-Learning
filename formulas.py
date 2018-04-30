# coding: utf-8
import numpy as np

def softmax(x):
    """use softmax(x) = softmax(x - max(x)) for x too large
    """
    x_T = np.transpose(x)
    x_T -= np.max(x_T, axis=0)
    y = (np.exp(x_T) / np.sum(np.exp(x_T), axis=0)).T
    return y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


import scipy as sp
def logloss(real, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(real*sp.log(pred) +
             sp.subtract(1, real)*sp.log(sp.subtract(1, pred)))
    ll = ll * 1.0/len(real)
    return ll


def auc(preds, labels):
    def tied_rank(x):
        sorted_x = sorted(zip(x, range(len(x))))
        r = [0 for k in x]
        cur_val = sorted_x[0][0]
        last_rank = 0
        for i in range(len(sorted_x)):
            if cur_val != sorted_x[i][0]:
                cur_val = sorted_x[i][0]
                for j in range(last_rank, i):
                    r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
                last_rank = i
            if i == len(sorted_x) - 1:
                for j in range(last_rank, i + 1):
                    r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
        return r
    r = tied_rank(preds)
    num_pos = len([0 for x in labels if x == 1])
    num_neg = len(labels) - num_pos
    sum_pos = sum([r[i] for i in range(len(r)) if labels[i] == 1])
    auc = (sum_pos - num_pos*(num_pos + 1)/2.0) / (num_neg*num_pos)
    return auc


if __name__ == "__main__":
    preds = [0.4, 0.6, 0.1, 0.8]
    labels = [1, 0, 0, 1]
    print(logloss(labels, preds))
    print(auc(preds, labels))

# coding = utf8
"""
    Naive Bayes Classifier
    
    @author: Shangru Zhong
"""
import numpy as np
from collections import defaultdict

class NaiveBayes(object):
    
    def __init__(self):
        self.prior = defaultdict(float)
        self.condi = []
        
    def fit(self, X, y):
        m, n = X.shape
        
        X[:, 1] = self.str2int(X[:, 1])
        # prior
        for y_val in y: 
            self.prior[y_val] += 1.0/len(y)
        # conditional
        for j in xrange(n):
            condi = defaultdict(lambda : defaultdict(int))
            for i in xrange(m):
                condi[y[i]][X[i, j]] += 1.0/(self.prior[y[i]]*len(y))
            self.condi.append(condi)
        return
    
    def predict(self, X):
        m, n = X.shape
        X[:, 1] = self.str2int(X[:, 1])
        
        pred = np.zeros(m)
        for i in xrange(m):
            post = defaultdict(float)
            for label, p in self.prior.iteritems():
                tmp = self.prior[label]
                for j in xrange(n):
                    tmp *= self.condi[j][label][X[i][j]]
                post[label] = tmp
            pred[i] = (self.dict_sort(post))[0][0]
        return pred
        
    def str2int(self, X1):
        try:
            trans = {'S':1, 'M':2, 'L':3}
            return [trans[val] for val in X1]
        except:
            return X1
        
    def dict_sort(self, d):
        return sorted(d.items(), key=lambda x:x[1], reverse=True)

if __name__ == "__main__":
  # test case in Hang Li's book (pp 50)
    X = np.array([[1, 'S'],
              [1, 'M'],
              [1, 'M'],
              [1, 'S'],
              [1, 'S'],
              [2, 'S'],
              [2, 'M'],
              [2, 'M'],
              [2, 'L'],
              [2, 'L'],
              [3, 'L'],
              [3, 'M'],
              [3, 'M'],
              [3, 'L'],
              [3, 'L']]
              )
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    model = NaiveBayes()
    model.fit(X, y)
    print "naive bayes model\nprior: \n{} \ncondition: \n{}".format(model.prior, model.condi)
    print "prediction: {}".format(model.predict(np.array([[2, 'S'], [3, 'L']])))
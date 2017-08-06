# coding=utf-8
"""
	K-means Clustering
	
	@author: Shangru 
	@2015/6/22
"""
import random
import scipy.io as sio
import numpy as np
#data m*n
#rows = len(data) m
#cols = len(data[0]) n
#center K*n

# Initialize K random centers(K*n)
def initial(data,K):
    center = [[0 for col in range(len(data[0]))] for row in range(K)]
    for i in range(K):
        randIndex = random.randint(1,len(data))
        center[i] = data[randIndex]
    return center

# Assign cluster for each data example 
def findClosestCenter(data,center):
    K = len(center) # num of clusters
    dist = [0]*K
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(K):
                dist[k] = dist[k] + (data[i][j]-center[k][j])**2
            idx[i] = dist.index(min(dist)) # storing cluster of each example
    return idx

# Move center of clusters
def computeCenter(data,idx,K):
    m = len(data)
    n = len(data[0])
    center = [[0 for col in range(n)] for row in range(K)]
    count = 0
    s = [0]*n
    for k in range(K):
        for j in range(m):
            if idx[j] == k:
                for i in range(n):
                    s[i] = s[i]+data[j][i]
                count = count+1
        center[k] = np.multiply(s, 1.0/count)
        count = 0
        s = [0]*n
    return center

if __name__ == "__main__":
    path = './dataset/clustering.mat'
    data = sio.loadmat(path)
    X = data['X']
    m, n = X.shape
    K = 3
    iters = 10
    center = initial(X, K)
    idx = [0]*m
    for i in range(iters):
        idx = findClosestCenter(X, center)
        center = computeCenter(X, idx, K)

    for p in range(m):
        print idx[p]
    
    

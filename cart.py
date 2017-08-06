# coding = utf-8
"""
	Classification And Regression Tree
	regression criterion: least squre

	@author: Shangru Zhong
"""
import numpy as np

class TreeNode(object):
    def __init__(self, feat=None, split=None):
        self.feat = feat
        self.split = split
        self.error = None
        self.tol = 0.01
        self.left = None
        self.right = None

class RegressionTree(object):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def create_tree(self, Xy):
        best_feat, best_split, error = self.best_split(Xy)
        if best_feat == None:
            return best_split
        root = TreeNode(best_feat, best_split) 
        if error < root.tol: # stop spliting
            return root 
        Xy_left = Xy[Xy[:, best_feat] <= best_split]
        Xy_right = Xy[Xy[:, best_feat] > best_split]   
        root.left = self.create_tree(Xy_left)
        root.right = self.create_tree(Xy_right)
        return root
    
    def best_split(self, Xy):
        m, n = Xy.shape
        best_feat = None
        best_split = None
        error = 1e9
        for j in xrange(n - 1): # for each feature
            print "current feature: {}".format(j)
            feat_vals = sorted(list(set(Xy[:, j])))
            feat_splits = [(feat_vals[i] + feat_vals[i + 1]) / 2 \
                         for i in xrange(len(feat_vals) - 1)]
            print "feature split: {}".format(feat_splits)
            feat_error = 1e9
            for s in feat_splits:
                R1 = Xy[Xy[:, j] <= s]
                R2 = Xy[Xy[:, j] > s]
                c1 = np.mean(R1[:, -1])
                c2 = np.mean(R2[:, -1])
                split_error = sum((R1[:, -1] - c1)**2) + sum((R2[:, -1] - c2)**2)
                print "error of split {} : {}".format(s, split_error)
                if split_error < feat_error:
                    feat_error = split_error
                    best_split = s
                
            if feat_error < error:
                error = feat_error
                best_feat = j
            print "best feature: {}, best split: {}, error: {}".format(best_feat, best_split, error)
        return best_feat, best_split, error       

if __name__ == "__main__":
	# test case in Hang Li's book (pp 149)
	X = np.array([np.arange(1, 11)])
	y = np.array([[5.56, 5.7, 5.92, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05]])
	Xy = np.concatenate((X.T, y.T), axis=1)

	model = RegressionTree()
	model.best_split(Xy)

	# # solution of problem 5.2 in Hang Li's book (pp 75)
	X = np.array([np.arange(1, 11)])
	y = np.array([[4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9.0]])
	Xy = np.concatenate((X.T, y.T), axis=1)

	model = RegressionTree()
	root = model.create_tree(Xy)
	print root.feat, root.split

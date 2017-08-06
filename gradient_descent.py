# coding: utf-8
"""
    Logistic Regression with Gradient Descent

    @author: Shangru
"""

import numpy as np  
import random

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):  
    converged = False  
    iter = 0  
    m = x.shape[0] # number of samples  
    # initial theta  
    t0 = np.random.random()  
    t1 = np.random.random()  

    # total error, J(theta)  
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])  
   
    # Iterate Loop  
    while not converged: 
    # for each training sample, compute the gradient (d/d_theta j(theta))  
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)])   
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])  
        # update the theta_temp  
        temp0 = t0 - alpha * grad0  
        temp1 = t1 - alpha * grad1  
       
        # update theta  
        t0 = temp0  
        t1 = temp1  
   
        # mean squared error  
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )   
   
        if abs(J-e) <= ep:  
            print 'Converged, iterations: ', iter, '!!!'  
            converged = True  
       
        J = e   # update error   
        iter += 1  # update iter  
        if iter == max_iter:  
            print 'Max interactions exceeded!'  
            converged = True  
    return t0,t1 

if __name__ == "__main__":
    alpha = 0.0001
    x = np.array([i for i in xrange(50)])
    y = np.array([2*i+3 for i in xrange(50)])
    print gradient_descent(alpha, x, y)
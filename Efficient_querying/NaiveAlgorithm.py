"""
Created on Sun Dec 28 2014
Last update: Sun Dec 28 2014

@author: Michiel Stock
michielfmstock@gmail.com

Basic implementation of a naive algorithm for efficient querying
"""

from collections import deque
from scipy import sparse
from sort_algorithms import argsort, quicksort, poor_mans_matrix_mult, poor_mans_dot


class NaiveAlgorithmSparse:
    def __init__(self, Y, init = False):
        '''
        Init 
        '''
        self.Y = Y.tocsr()
        self.Ni, self.K = Y.shape
        self.steps_taken = [] #number of steps the algororithm takes for one query
        self.calculated = [] #number of scores for one query
        if init:
            self.init_sorted_lists()
            
                
    def init_sorted_lists(self):
        """
        Makes for each latent feature a list of the sorted indices for each item
        No sorting phase in this algorithm
        """
        print 'This does nothing...'
                
    def bestNlist(self, X, N = 10):
        '''
        Calculates at least the top-N elements for each user 
        
        INPUTS
            X: feature matrix for the different users
            N: number of top items to find
        '''
        return [self.bestN(x_u, N) for x_u in X]
        
        
    def bestN(self, x_u, N):
        '''
        `Threshold algorithm for one user u
        
        INPUTS
            x_u: features of user u
            N: number of top items to find
            
        OUTPUT
            list_u: a dictionary containing the items with their scores, contains at least the N items with the best scores
        '''
        feature_ind = range(self.K)
        x_u = x_u.tocoo()#only for when x_u differs from 0
        nonneg_x = x_u.col
        x_dense = x_u.T.todense()
        S_values = []
        steps = 0
        n_items = 0
        for it in xrange(self.Ni):
            steps += 1
            y = self.Y.getrow(it)
            S = (y.multiply(x_u)).sum()
            if n_items < N:
                S_values.append( (S, it) )
                n_items += 1
                lowerS = min(S_values)[0]
            elif S > lowerS:
                replace_index = S_values.index(min(S_values))
                S_values[replace_index] = (S, it)
                lowerS = min(S_values)[0]
        self.steps_taken.append(steps)
        self.calculated.append(steps)
        return S_values
        
class NaiveAlgorithm:
    def __init__(self, Y, init = False):
        '''
        Init 
        '''
        self.Y = Y
        self.Ni = len(Y)
        self.K = len(Y[0])
        self.steps_taken = [] #number of steps the algororithm takes for one query
        self.calculated = [] #number of scores for one query
        if init:
            self.init_sorted_lists()
            
                
    def init_sorted_lists(self):
        """
        Makes for each latent feature a list of the sorted indices for each item
        No sorting phase in this algorithm
        """
        print 'This does nothing...'
                
    def bestNlist(self, X, N = 10):
        '''
        Calculates at least the top-N elements for each user 
        
        INPUTS
            X: feature matrix for the different users
            N: number of top items to find
        '''
        return [self.bestN(x_u, N) for x_u in X]
        
    def repairL(self, stepsBack, reversedks):
        '''
        After running Fagin_u, this repairs the matrix of sorted indices
        '''
        for Lk in self.L:
            Lk.rotate(-stepsBack)
        for k in reversedks:
            self.L[k].reverse()
        
    def bestN(self, x_u, N):
        '''
        `Threshold algorithm for one user u
        
        INPUTS
            x_u: features of user u
            N: number of top items to find
            
        OUTPUT
            list_u: a dictionary containing the items with their scores, contains at least the N items with the best scores
        '''
        x_u = deque(x_u)
        feature_ind = range(self.K)
        S_values = []
        steps = 0
        n_items = 0
        for it in xrange(self.Ni):
            steps += 1
            S = poor_mans_dot(self.Y[it], x_u)
            if n_items < N:
                S_values.append( (S, it) )
                n_items += 1
                lowerS = min(S_values)[0]
            elif S > lowerS:
                replace_index = S_values.index(min(S_values))
                S_values[replace_index] = (S, it)
                lowerS = min(S_values)[0]
        self.steps_taken.append(steps)
        self.calculated.append(steps)
        return S_values
       

if __name__=="__main__":

    
    import time
    import random as rd
    import numpy as np
    
    rd.seed(99)

    K = 10
    N = 5
    nx = 100
    ny = 5000
    X = np.random.randn(nx, K).tolist()
    Y = np.random.randn(ny, K).tolist()
    
    start = time.clock()
    
    
    NA = NaiveAlgorithm(Y)
    
    
    NA.init_sorted_lists()
    
    Fagin_Sort_Time = time.clock()
    
    S = NA.bestNlist(X, 5)
    
    Fagin_Mult_Time = time.clock()
    

    Exact_Mult_Time = time.clock()
    
    Smax_exact = np.dot(X, np.array(Y).T).max(1)
    Exact_Sort_Time = time.clock()
    
    for i in range(nx):
        print (Smax_exact[i] - max(S[i])[0])**2



    print 'Time to sort in Fagin', Fagin_Sort_Time - start
    print 'Time to multiply in Fagin', Fagin_Mult_Time - Fagin_Sort_Time
    print 'Time to multiply in naive algorithm', Exact_Mult_Time - Fagin_Sort_Time
    print 'Time to sort in naive algorithm', Exact_Sort_Time - Exact_Mult_Time
    
    print 'Average number of steps in Fagin', np.mean(NA.steps_taken)
    
    
    
    from sklearn.datasets import load_boston

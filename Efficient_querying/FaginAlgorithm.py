"""
Created on Sun Dec 28 2014
Last update: Sun Dec 28 2014

@author: Michiel Stock
michielfmstock@gmail.com

Basic implementation of Fagin's algorithm for efficient querying
"""

from collections import deque
from sort_algorithms import argsort, quicksort, poor_mans_matrix_mult, poor_mans_dot
from scipy import sparse
import numpy as np


class FaginAlgorithmSparse:
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
        """
        self.Y = self.Y.tocoo()
        self.L = {k:[] for k in range(self.K)}
        [self.L[self.Y.col[i]].append( (self.Y.data[i], self.Y.row[i])) for i in xrange(self.Y.nnz) ]
        [self.L[k].sort() for k in range(self.K)]
        [self.L[k].reverse() for k in range(self.K)]
  
           
    def bestNlist(self, X, N = 10):
        '''
        Calculates at least the top-N elements for each user 
        
        INPUTS
            X: feature matrix for the different users: as list of lists
            N: number of top items to find
        '''
        return [self.bestN(x_u, N) for x_u in X]
        
    def repairL(self, stepsBack, reversedks):
        '''
        After running Fagin_u, this repairs the matrix of sorted indices
        '''
        for Lk in self.L:
            if len(Lk) > 0:
                Lk.rotate(-stepsBack)
        #for k in reversedks:
        #    self.L[k].reverse()
        
    def bestN(self, x_u, N):
        '''
        Fagin algorithm for one user u, warning, only for sparse, non-negative matrices!
        
        INPUTS
            x_u: features of user u
            N: number of top items to find
            
        OUTPUT
            list_u: a dictionary containing the items with their scores, contains at least the N items with the best scores
        '''
        item_list = set([])
        x_u = x_u.tocoo()#only for when x_u differs from 0
        nonneg_x = x_u.col
        x_dense = x_u.T.todense()
        lower_bound = -1e5
        upper_bound = 1e5
        terminate = False # check if any objects in Y
        if len(nonneg_x) == 0: #recognise when no matches can be found
            terminate = True
        list_u = {}
        feature_ind = range(self.K)
        n_items = 0
        S_values = []
        steps = 0
        n_nonneg = [len(self.L[k]) for k in feature_ind]
        occurences = np.zeros(self.Ni, dtype = int)
        it_thr = {}
        unionset = set([])
        finished = False
        while len(it_thr) < N and len(unionset) < self.Ni and not finished:
            for k in nonneg_x:
                finished = True #if no more items with nonzero value: stop
                if n_nonneg[k] > steps:
                    pop_it = self.L[k][steps][1]
                    occurences[pop_it] += 1
                    unionset.add(pop_it)
                    finished = False
                    if occurences[pop_it] >= self.K:
                        it_thr[pop_it] = occurences[pop_it]
            steps += 1
        for it in unionset:
            y = self.Y.getrow(it)
            S = y.multiply(x_u).sum()
            if n_items < N:
                S_values.append( (S, it) )
                lowerS = min(S_values)[0]
            elif S > lowerS:
                replace_index = S_values.index(min(S_values))
                S_values[replace_index] = (S, it)
                lowerS = min(S_values)[0]
            n_items += 1
        self.steps_taken.append(steps)
        self.calculated.append(n_items)
        return S_values



        
class FaginAlgorithm:
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
            
                
    def init_sorted_lists(self, npsort = False):
        """
        Makes for each latent feature a list of the sorted indices for each item
        """
        if not npsort:
            self.L = [ deque(argsort(l)) for l in map(list, zip(*self.Y)) ]
        else:
            self.L = [ deque(np.argsort(l)) for l in map(list, zip(*self.Y)) ]
                
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
        reversed = [k for k in range(self.K) if x_u[k] < 0]
        [(self.L[k]).reverse() for k in reversed]
        list_u = {}
        feature_ind = range(self.K)
        n_items = 0
        S_values = []
        steps = 0
        occurences = np.zeros(self.Ni, dtype = int)
        it_thr = {}
        unionset = set([])
        while len(it_thr) < N and len(unionset) < self.Ni:
            steps += 1
            for k in feature_ind:
                pop_it = self.L[k].pop()
                self.L[k].appendleft(pop_it)
                occurences[pop_it] += 1
                unionset.add(pop_it)
                if occurences[pop_it] >= self.K:
                    it_thr[pop_it] = occurences[pop_it]
        for it in unionset:
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
        self.calculated.append(len(unionset))
        self.repairL( steps, reversed)
        return S_values
       

if __name__=="__main__":

    

    
    import time
    import random as rd
    import numpy as np
    from sklearn.preprocessing import normalize
    
    rd.seed(99)
    
    testNormal = True
    testSparse = False
    
    
    if testNormal: 
	    K = 10
	    N = 5
	    nx = 100
	    ny = 5000
	    X = np.random.randn(nx, K).tolist()
	    Y = np.random.randn(ny, K).tolist()
	    
	    start = time.clock()
	    
	    
	    TA = FaginAlgorithm(Y)
	    
	    
	    TA.init_sorted_lists()
	    
	    Fagin_Sort_Time = time.clock()
	    
	    S = TA.bestNlist(X, 5)
	    
	    Fagin_Mult_Time = time.clock()
	    
	
	    Exact_Mult_Time = time.clock()
	    
	    Smax_exact = np.dot(X, np.array(Y).T).max(1)
	    Exact_Sort_Time = time.clock()
	    
	    for i in range(nx):
	        print (Smax_exact[i] - max(S[i])[0])**2
	
	
	
	    print 'Time to sort in Fagin', Fagin_Sort_Time - start
	    print 'Time to multiply in Fagin', Fagin_Mult_Time - Fagin_Sort_Time
	    print 'Time to multiply in naive algorithm', Exact_Mult_Time - Fagin_Sort_Time
	    print 'Time to run naive algorithm', Exact_Sort_Time - Exact_Mult_Time
	    
	    print 'Average number of steps in Fagin', np.mean(TA.steps_taken)
	    
    if testSparse:
        density = 0.001
        nx = 10000
        ny = 10000
        Y = sparse.rand(ny, nx, density)
        Y = normalize(Y, axis=1, norm  = 'l2')
        TAS = FaginAlgorithmSparse(Y)
        TAS.init_sorted_lists()
        Y.tocsr()
        qids = range(ny)
        rd.shuffle(qids)
        Scores = []
        for i in qids[:100]:
            S = TAS.bestN(Y.getrow(i), 5)
            print S
        #print Scores

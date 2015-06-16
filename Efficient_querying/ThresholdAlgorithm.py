"""
Created on Sun Dec 28 2014
Last update: Sun Dec 28 2014

@author: Michiel Stock
michielfmstock@gmail.com

Basic implementation of the threshold algorithm for efficient querying
"""

from collections import deque
from sort_algorithms import argsort, quicksort, poor_mans_matrix_mult, poor_mans_dot
from scipy import sparse
from itertools import izip
import numpy as np
        
class ThresholdAlgorithmSparse:
    def __init__(self, Y, init = False):
        '''
        Init 
        '''
        self.Y = Y
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
        '''
        self.L = [filter(lambda t:t[1]==k, tuples) for k in range(self.K)]
        self.L = [sorted(Lk, key = lambda t:t[2]) for Lk in L if len(Lk) >0]
        for k in xrange(self.K):
            l = zip(self.Y[:,k].data, self.Y[:,k].row)
            l.sort()
            self.L.append([ tup for tup in l ])
        self.Y.tocsr()
        '''
           
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
        Threshold algorithm for one user u, warning, only for sparse, non-negative matrices!
        
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
        if len(nonneg_x) == 0: #recognise when no matches can be found
            lower_bound = 1
            upper_bound = 0
        reversed = []#[k for k in range(self.K) if x_u[k] < 0]
        list_u = {}
        feature_ind = range(self.K)
        n_items = 0
        S_values = []
        steps = 0
        n_nonneg = [len(self.L[k]) for k in feature_ind]
        terminate = False # check if any objects in Y
        while lower_bound < upper_bound and not terminate:
            upper_bound = 0.0
            items = deque([])
            s2 = deque([])
            terminate = True
            for k in nonneg_x:
                if n_nonneg[k] > steps:
                    terminate = False
                    tup = self.L[k][steps]
                    it = tup[1]
                    items.append(it)
                    s2.append(k)
                    upper_bound += x_dense[k]*tup[0]
            new_items = set(items) - item_list
            item_list.update(items)
            for it in items:
                if it in new_items:
                    y = self.Y.getrow(it)
                    S = y.multiply(x_u).sum()
                    if n_items < N:
                        S_values.append( (S, it) ) #if less than N elements, add to list
                        lower_bound = min(S_values)[0]
                    elif S > lower_bound:
                        replace_index = S_values.index(min(S_values))
                        S_values[replace_index] = (S, it) 
                        lower_bound = min(S_values)[0]
                    n_items += 1
            steps += 1  
        #self.repairL(steps, reversed)#restore the lists for future use
        self.steps_taken.append(steps)
        self.calculated.append(n_items)
        return S_values
        
class ThresholdAlgorithm:
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
        x_u = deque(x_u)
        item_list = set([])
        lower_bound = -1e5
        upper_bound = 1e5
        reversed = [k for k in range(self.K) if x_u[k] < 0]
        [(self.L[k]).reverse() for k in reversed]
        list_u = {}
        feature_ind = range(self.K)
        n_items = 0
        S_values = []
        steps = 0
        while lower_bound < upper_bound:
            upper_bound = 0.0
            items = [Lk.pop() for Lk in self.L]
            [(self.L[k]).appendleft(items[k]) for k in range(self.K)]
            new_items = list(set(items) - item_list)
            item_list.update( new_items )
            for k in feature_ind:
                item = items[k]
                upper_bound += self.Y[item][k]*x_u[k]
                if item in new_items:
                    S = poor_mans_dot(self.Y[item], x_u)
                    n_items += 1
                    if n_items < N:
                        S_values.append( (S, item) ) #if less than N elements, add to list
                        lower_bound = min(S_values)[0]
                    elif S > lower_bound:
                        replace_index = S_values.index(min(S_values))
                        S_values[replace_index] = (S, item) 
                        lower_bound = min(S_values)[0]
                        
            steps += 1
        self.repairL(steps, reversed)#restore the lists for future use
        self.steps_taken.append(steps)
        self.calculated.append(n_items)
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
	    N = 10
	    nx = 100
	    ny = 50000
	    X = np.random.randn(nx, K).tolist()
	    Y = np.random.randn(ny, K).tolist()
	    
	    start = time.clock()
	    
	    
	    TA = ThresholdAlgorithm(Y)
	    
	    
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
        ny = 10001
        D = sparse.rand(ny, nx, density)
        D = normalize(D, axis=1, norm  = 'l2') 
        
        
        TAS = ThresholdAlgorithmSparse(D)
        TAS.init_sorted_lists()
        D.tocsr()
        qids = range(ny)
        rd.shuffle(qids)
        Scores = []
        for i in qids[:100]:
            S = TAS.bestN(D.getrow(i), 5)
            print S
        #print Scores
	    
	    
	
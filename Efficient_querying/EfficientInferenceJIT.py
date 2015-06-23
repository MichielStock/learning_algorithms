"""
Created on Fri Jun 19 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of algorithms for the efficient inference of the top-K of
seperable linear models, made to be compatible with JIT
"""

from heaps import MaxHeap, MinHeap
from numba import jit
from time import time

@jit
def score_item(x, Y, item):
    """
    Calculates the score of an item
    """
    R = len(x)
    score = 0
    for r in range(R):
        score += x[r] * Y[item, r]
    return (score, item)

@jit
def partial_score_item(x, Y, item, partials, upper_bound, lower_bound, R):
    """
    Scores an item, but terminates when it becomes impossible to improve upon
    the lower bound
    """
    r = 0
    while upper_bound > lower_bound and r < R:
        upper_bound -= partials[r]
        upper_bound += x[r] * Y[item,r]
        r += 1
    if r < R:
        return False, r, None
    else:
        return True, r, (upper_bound, item)

#@jit
def get_top_naive(x, Y, topheap):
    M, R = Y.shape
    n_calculations = 0
    for item in range(M):
        score = score_item(x, Y, item)
        topheap.heapupdate(score)
        n_calculations += R
    return topheap, n_calculations

#@jit
def get_top_threshold(x, Y, topheap, sorted_lists, is_scored):
    M, R = Y.shape
    n_calculations = 0
    upper_bound = np.inf
    lower_bound = -np.inf
    pos = 0
    while upper_bound > lower_bound:
        upper_bound = 0
        for r in range(R):
            xr = x[r]
            if xr >= 0:
                item = sorted_lists[pos, r]
            else:
                item = sorted_lists[M - 1 - pos, r]
            if not is_scored[item]:
                score = score_item(x, Y, item)
                topheap.heapupdate(score)
                n_calculations += R
                is_scored[item] = 1
                lower_bound = topheap.peek()[0]
            upper_bound += x[r] * Y[item, r]
        pos += 1
    return topheap, n_calculations

#@jit
def get_top_threshold_partial(x, Y, topheap, sorted_lists, is_scored, partial_scores):
    M, R = Y.shape
    pos = 0
    for r in range(R):
        item = sorted_lists[pos, r]
        pr = x[r] * Y[item, r]
        partial_scores[r] = pr
    n_calculations = 0
    upper_bound = partial_scores.sum()
    lower_bound = -np.inf
    while upper_bound > lower_bound:
        for r in range(R):
            xr = x[r]
            if xr >= 0:
                item = sorted_lists[pos, r]
            else:
                item = sorted_lists[M - 1 - pos, r]
            pr = x[r] * Y[item, r]
            upper_bound -= partial_scores[r]
            partial_scores[r] = pr
            upper_bound += pr
            if not is_scored[item]:
                is_compl, n_calc, score = partial_score_item(x, Y, item,
                    partial_scores, upper_bound, lower_bound, R)
                if is_compl:
                    topheap.heapupdate(score)
                n_calculations += n_calc
                is_scored[item] = 1
                lower_bound = topheap.peek()[0]
        pos += 1
    return topheap, n_calculations

class TopKInference():
    """
    A module collecting different algorithms to find the top-K for a given
    query and SEP-LR model.
    This class is designed for dense matrices
    """
    def __init__(self, Y, initialize_lists=True):
        self.V = np.linalg.eigh(Y.T.dot(Y))[1]  # calculate eigenvectors
        self.Y = Y
        self.M, self.R = Y.shape
        if initialize_lists:
            self.initialize_sorted_lists()

    def initialize_sorted_lists(self):
        """
        Initializes the sorted lists
        """
        self.sorted_lists = (-self.Y).argsort(0)

    def get_top_K(self, queries, K=1, algorithm='threshold', profile=False):
        """
        Returns the top-K objects for a given query
        """
        n_calculations = []
        runtimes = []
        top_Ks = []
        partial_scores = np.zeros(self.R)
        for x_u in queries:
            topheap = MinHeap(-np.ones(K)*np.inf, np.arange(K))
            is_scored = np.zeros(self.M, dtype=int)
            t1 = time()
            if algorithm == 'partial_threshold':
                top_list, n_calc = get_top_threshold_partial(x_u, self.Y, topheap,
                        self.sorted_lists, is_scored, partial_scores)
            elif algorithm == 'threshold':
                top_list, n_calc = get_top_threshold(x_u, self.Y, topheap,
                        self.sorted_lists, is_scored)
            elif algorithm == 'naive':
                top_list, n_calc =  get_top_naive(x_u, self.Y, topheap)
            else:
                print 'Unknown algorithm selected...'
                raise KeyError
            t2 = time()
            top_Ks.append(top_list.listify())  # save top
            if profile:
                runtimes.append(t2 - t1)
                n_calculations.append(n_calc)
        if not profile:
            return top_Ks
        else:
            return top_Ks, n_calculations, runtimes

if __name__ == '__main__':

    import numpy as np

    # TESTING THE DENSE FRAMEWORK
    # ---------------------------

    R = 10
    n = 50000
    K = 3


    Y = np.random.randn(n, R)

    sorted_lists = (-Y).argsort(0)

    x = np.random.randn(R)**2

    top_5_naive = MinHeap(-np.ones(K)*np.inf, np.arange(K))
    top_5_naive, n_calc_naive = get_top_naive(x, Y, top_5_naive)

    top_5_thr = MinHeap(-np.ones(K)*np.inf, np.arange(K))
    is_scored = np.zeros(n, dtype=int)
    top_5_thr, n_calc_thr = get_top_threshold(x, Y, top_5_thr, sorted_lists, is_scored)

    partial_scores = np.zeros(R, dtype=float)
    top_5_thr_partial = MinHeap(-np.ones(K)*np.inf, np.arange(K))
    is_scored = np.zeros(n, dtype=int)
    top_5_thr_partial, n_calc_thr_part = get_top_threshold_partial(x, Y, top_5_thr_partial, sorted_lists, is_scored, partial_scores)


    # TESTING THE MODULE
    # -------------------

    queries = np.random.randn(10, R)**2

    inferer = TopKInference(Y)

    top_K_naive, n_calc_naive, runtimes_naive = inferer.get_top_K(queries, K=5, algorithm='naive', profile=True)
    top_K_thr, n_calc_thr, runtimes_thr = inferer.get_top_K(queries, K=5, algorithm='threshold', profile=True)
    top_K_partial, n_calc_partial, runtimes_partial = inferer.get_top_K(queries, K=5, algorithm='partial_threshold', profile=True)

    print 'Tested for data of size %s with R of %s' %(n, R)
    print 'Naive: %s calculations in %s seconds' %(np.mean(n_calc_naive), np.mean(runtimes_naive))
    print 'Threshold: %s calculations in %s seconds' %(np.mean(n_calc_thr), np.mean(runtimes_thr))
    print 'Partial threshold: %s calculations in %s seconds' %(np.mean(n_calc_partial), np.mean(runtimes_partial))
    print

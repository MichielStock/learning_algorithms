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

def score_item(x, Y, item):
    """
    Calculates the score of an item
    """
    R = len(x)
    score = 0
    for r in range(R):
        score += x[r] * Y[item, r]
    return (score, item)

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

def get_top_naive(x, Y, topheap):
    M, R = Y.shape
    n_calculations = 0
    for item in range(M):
        score = score_item(x, Y, item)
        topheap.heapupdate(score)
        n_calculations += R
    return topheap, n_calculations

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
                print score
                topheap.heapupdate(score)
                n_calculations += R
                is_scored[item] = 1
                lower_bound = topheap.peek()[0]
            upper_bound += x[r] * Y[item, r]
        pos += 1
    return topheap, n_calculations

if __name__ == '__main__':

    import numpy as np

    # TESTING THE DENSE FRAMEWORK
    # ---------------------------

    R = 10
    n = 50000
    K = 5


    Y = np.random.rand(n, R)

    sorted_lists = (-Y).argsort(0)

    x = np.random.randn(R)

    top_5_naive = MinHeap(-np.ones(K)*np.inf, np.arange(K))
    top_5_naive, n_cal_naive = get_top_naive(x, Y, top_5_naive)

    top_5_thr = MinHeap(-np.ones(K)*np.inf, np.arange(K))
    is_scored = np.zeros(n, dtype=int)
    top_5_thr, n_calc_thr = get_top_threshold(x, Y, top_5_thr, sorted_lists, is_scored)

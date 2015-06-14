"""
Created on Tue Jun 9 2015
Last update: Sat Jun 13 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of algorithms for the efficient inference of the top-K of
seperable linear models
"""

from numpy import dot
from heapq import heapify, heappop, heappush, heapreplace
from time import time
from numba import jit

def calculate_partial_score(r, pos, xi, Y, sorted_lists):
    """
    calculates a partial score for the position in the list
    """
    indice = sorted_lists[conditionalposition(pos, xi), r]
    return xi * Y[indice, r]

def conditionalposition(pos, xi):
    """
    returns position such that:
            pos if xi < 0
            - pos - 1 if  xi >= 0
    """
    if xi < 0:
        return pos
    else:
        return - pos - 1

class TopKInference():
    """
    A module collecting different algorithms to find the top-K for a given
    query and SEP-LR model.
    This class is designed for dense matrices
    """
    def __init__(self, Y, initialize_lists=False):
        self.Y = Y
        self.M, self.R = Y.shape
        if initialize_lists:
            self.initialize_sorted_lists()

    def initialize_sorted_lists(self):
        """
        Initializes the sorted lists
        """
        self.sorted_lists = self.Y.argsort(0)

    def get_top_K(self, queries, K=1, algorithm='enhanced', profile=False):
        """
        Returns the top-K objects for a given query
        """
        n_scores_calc = []
        runtimes = []
        top_Ks = []
        for x_u in queries:
            if algorithm=='enhanced':
                top_list, n_items_scored, time = self.get_top_K_threshold_enhanced(\
                    x_u, K, count_calculations=True)
            elif algorithm=='threshold':
                top_list, n_items_scored, time = self.get_top_K_threshold(\
                    x_u, K, count_calculations=True)
            elif algorithm=='naive':
                top_list, n_items_scored, time = self.get_top_K_naive(\
                    x_u, K, count_calculations=True)
            else:
                print 'Unknown algorithm selected...'
                raise KeyError
            top_Ks.append(top_list)
            if profile:
                runtimes.append(time)
                n_scores_calc.append(n_items_scored)
        if not profile:
            return top_Ks
        else:
            return top_Ks, n_scores_calc, runtimes

    """
    def score_item(self, x_u, indice):
        return (dot(self.Y[indice], x_u), indice)
    """

    def score_item(self, x_u, indice):
        result = 0.0
        for i, xi in enumerate(x_u):
            result += xi * self.Y[indice, i]
        return (result, indice)

    def update_top_list(self, top_list, new_scored_item, K, n_scored):
        """
        Updates the top-K list with a new scored item
        """
        if n_scored < K:
            heappush(top_list, new_scored_item )
        elif new_scored_item[0] > top_list[0][0]:
            # case that the new item is better than the worst in the list
            heapreplace(top_list, new_scored_item)  # replace the worst with the current
        # returns nothing, processes the list

    def get_top_K_naive(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K for a given query by naively scoring all the items
        """
        t0 = time()
        top_list = [(-1e10,) for i in range(K)]
        n_items_scored = 0
        for indice in range(self.M):
            new_scored_item = self.score_item(x_u, indice)
            if top_list[0][0] < new_scored_item[0]:
                heapreplace(top_list, new_scored_item)
            n_items_scored += 1
        top_list.sort()
        t1 = time()
        if count_calculations:
            return top_list, n_items_scored, t1 - t0
        else:
            return top_list

    def get_top_K_fagin(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using Fagin's algorithm
        """
        top_list = []
        n_items_scored = 0

        # fill in
        if count_calculations:
            return top_list, count_calculations
        else:
            return top_list

    def get_top_K_threshold(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using the threshold algorithm
        """
        t0 = time()
        top_list = [(-1e10,) for i in range(K)]
        n_items_scored = 0
        neg_elements_query = set([i for i, el in enumerate(x_u) if el < 0])
        non_zero_elements_query = [i for i, el in enumerate(x_u) if el != 0]
        scored = set([])
        upper_bound = 1
        depth = 0
        while upper_bound > top_list[0][0]:
            upper_bound = 0
            for r in non_zero_elements_query:
                if r in neg_elements_query:
                    item = self.sorted_lists[depth, r]  # negative, so start from
                            # items with the LOWEST score for this item
                else:
                    item = self.sorted_lists[-(depth+1), r]
                    # update upper bound
                upper_bound += self.Y[item, r] * x_u[r]
                if item not in scored:
                    new_scored_item = self.score_item(x_u, item)
                    if top_list[0][0] < new_scored_item[0]:
                        heapreplace(top_list, new_scored_item)
                    n_items_scored += 1
                    scored.add(item)
            depth += 1
        top_list.sort()
        t1 = time()
        if count_calculations:
            return top_list, n_items_scored, t1 - t0
        else:
            return top_list

    def get_top_K_threshold_enhanced(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using the modified threshold algorithm
        """
        t0 = time()
        top_list = [(-1e10,) for i in range(K)]
        n_items_scored = 0
        # initiate list with rules
        # contains tuples with
        # (-paritial score, xi, r, position_sorted_list, decr/incr sorted)
        # note negetive partial score for the heap!
        query_info_list = [(-calculate_partial_score(r, 0, xi, self.Y, self.sorted_lists),
                        xi,
                        r,
                        0 if xi < 0 else -1,
                        1 if xi < 0 else -1)\
                for r, xi in enumerate(x_u) if xi != 0]
        heapify(query_info_list)  # turn in a heap in O(R) time
        scored = set([])
        #  we start with the upper bound which we update iteratively
        upper_bound = sum([-x[0] for x in query_info_list])
        while upper_bound > top_list[0][0]:
            partial_score, xi, r, pos, pos_action = heappop(query_info_list)
            # get item
            item = self.sorted_lists[pos, r]
            # score item
            if item not in scored:
                new_scored_item = self.score_item(x_u, item)
                if top_list[0][0] < new_scored_item[0]:
                    heapreplace(top_list, new_scored_item)
                n_items_scored += 1
                scored.add(item)
            # update position for this list
            pos += pos_action
            # update the upper bound
            upper_bound += partial_score  # remove previous partial score (neg)
            partial_score = xi * self.Y[item, r]  # get new partial score
            upper_bound += partial_score
            # update the rule list
            heappush(query_info_list, (-partial_score, xi, r, pos, pos_action))
        top_list.sort()
        t1 = time()
        if count_calculations:
            return top_list, n_items_scored, t1 - t0
        else:
            return top_list

class TopKInferenceSparse(TopKInference):
    """
    A module collecting different algorithms to find the top-K for a given
    query and SEP-LR model.
    This class is designed for dense matrices, for which the elements are
    positive.
    """
    def initialize_sorted_lists(self):
        """
        Makes for each latent feature a list of the sorted indices for each item
        """
        self.Y = self.Y.tocoo()
        self.sorted_lists = {r:[] for r in range(self.R)}
        [self.sorted_lists[self.Y.col[i]].append( (self.Y.data[i], self.Y.row[i])) for i in xrange(self.Y.nnz) ]
        [self.sorted_lists[r].sort() for r in range(self.R)]
        [self.sorted_lists[r].reverse() for r in range(self.R)]
        self.Y = self.Y.tocsr()

    def score_item(self, x_u, indice):
        """
        Scores an item (sparse vector multiplication)
        """
        return ((self.Y[indice] * x_u.tocsc().T)[0,0], indice)

    def get_top_K_threshold(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using the threshold algorithm, suited for sparse data
        """
        t0 = time()
        x_u = x_u.tocoo()
        top_list = [(-1e10,) for i in range(K)]
        n_items_scored = 0
        non_zero_elements_query = [(xi, i) for i, xi in zip(x_u.col, x_u.data)]
        scored = set([])
        depth = 0
        upper_bound = 1e10
        if len(non_zero_elements_query) == 0:
            upper_bound = -1  # break when no x
        while upper_bound > top_list[0][0]:
            upper_bound = 0
            for xi, r in non_zero_elements_query:
                if len(self.sorted_lists[r]) > depth:
                    yir, item = self.sorted_lists[r][depth]
                    upper_bound += xi * yir
                    if item not in scored:
                        new_scored_item = self.score_item(x_u, item)
                        scored.add(item)
                        if new_scored_item[0] > top_list[0][0]:
                            heapreplace(top_list, new_scored_item)
                        n_items_scored += 1
            depth += 1
        top_list.sort()
        t1 = time()
        if count_calculations:
            return top_list, n_items_scored, t1 - t0
        else:
            return top_list

    def get_top_K_threshold_enhanced(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using the modified threshold algorithm, suited for sparse data
        """
        t0 = time()
        x_u = x_u.tocoo()
        top_list = [(-1e10,) for i in range(K)]
        n_items_scored = 0
        # initiate list with rules
        # contains tuples with
        # (-paritial score, xi, r, position_sorted_list)
        # note negetive partial score for the heap!
        query_info_list = [( - xi * self.sorted_lists[r][0][0],
                        xi,
                        r,
                        0)\
                for r, xi in zip(x_u.col, x_u.data)\
                if len(self.sorted_lists[r][0]) > 0]
        heapify(query_info_list)  # turn in a heap in O(R) time
        scored = set([])
        #  we start with the upper bound which we update iteratively
        upper_bound = sum([-x[0] for x in query_info_list])
        lower_bound = -1e100
        while upper_bound > top_list[0][0]:
            if len(query_info_list) == 0:
                break
            partial_score, xi, r, pos = heappop(query_info_list)
            # get item
            item = self.sorted_lists[r][pos][1]
            # score item
            if item not in scored:
                new_scored_item = self.score_item(x_u, item)
                if new_scored_item[0] > top_list[0][0]:
                    heapreplace(top_list, new_scored_item)
                n_items_scored += 1
                scored.add(item)
            # update position for this list
            pos += 1
            # update the upper bound
            upper_bound += partial_score  # remove previous partial score (neg)
            # update the rule list
            if pos < len(self.sorted_lists[r]):  # if there are still non-zero elements...
                partial_score = xi * self.sorted_lists[r][pos][0]  # get new partial score
                upper_bound += partial_score
                heappush(query_info_list, (-partial_score, xi, r, pos))
        top_list.sort()
        t1 = time()
        if count_calculations:
            return top_list, n_items_scored, t1 - t0
        else:
            return top_list

if __name__ == '__main__':

    import numpy as np
    from time import clock

    # TESTING THE DENSE FRAMEWORK
    # ---------------------------

    R = 10
    n = 1000000
    K = 5

    W = np.random.rand(n, R)**2

    inferer = TopKInference(W)

    x = np.random.randn(R)**2

    top_5_list_naive, n_scored_naive, runtime_naive = inferer.get_top_K_naive(x, K, True)

    inferer.initialize_sorted_lists()

    top_5_list_threshold, n_scored_threshold, runtime_thr = inferer.get_top_K_threshold(x, K, True)

    top_5_list_threshold_enh, n_scored_threshold_enh, runtime_enh = inferer.get_top_K_threshold_enhanced(x, K, True)


    print 'Tested for data of size %s with R of %s' %(n, R)
    print 'Naive: %s calculations in %s seconds' %(n_scored_naive, runtime_naive)
    print 'Threshold: %s calculations in %s seconds' %(n_scored_threshold, runtime_thr)
    print 'Enhanced threshold: %s calculations in %s seconds' %(n_scored_threshold_enh, runtime_enh)
    print

    # TESTING THE SPARSE FRAMEWORK
    # ----------------------------

    from scipy import sparse

    R = 1000
    n = 10000
    K = 5

    Y = sparse.rand(n, R, density=0.001)

    sparse_inferer = TopKInferenceSparse(Y)
    sparse_inferer.initialize_sorted_lists()

    x_u = sparse.rand(1,R)

    top_5_list_naive, n_scored_naive, runtime_naive = sparse_inferer.get_top_K_naive(x_u
, K, True)

    top_5_list_threshold, n_scored_threshold, runtime_thr = sparse_inferer.get_top_K_threshold(x_u , K, True)

    top_5_list_thr_enh, n_scored_thr_enh, runtime_thr_enh = sparse_inferer.get_top_K_threshold_enhanced(x_u, K, True)


    print 'Tested for SPARSE data of size %s with R of %s' %(n, R)
    print 'Naive: %s calculations in %s seconds' %(n_scored_naive, runtime_naive)
    print 'Threshold: %s calculations in %s seconds' %(n_scored_threshold, runtime_thr)
    print 'Enhanced threshold: %s calculations in %s seconds' %(n_scored_thr_enh, runtime_thr_enh)
    print

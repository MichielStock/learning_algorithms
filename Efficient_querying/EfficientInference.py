"""
Created on Tue Jun 9 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of algorithms for the efficient inference of the top-K of
seperable linear models
"""

from numpy import dot
from numba import jit

def calculate_partial_score(r, pos, xi, Y, sorted_lists):
    """
    calculates a partial score for the position in the list
    """
    indice = sorted_lists[conditionalposition(pos, xi), r]
    return xi * Y[indice, r]

def swap(l, i, j):
    # swaps two elements on a postion of a list
    temp = l[i]
    l[i] = l[j]
    l[j] = temp

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

def cure_partial_ordered_list(posset):
    """
    Sorts a partial ordered list for which only the first element is out of
    place
    """
    n = len(posset)
    for i in range(n-1):
        if posset[i][0] < posset[i+1][0]: # elements are not ordered
            swap(posset, i, i+1)
        else:
            break

class TopKInference():
    """
    A module collecting different algorithms to find the top-K for a given
    query and SEP-LR model.
    This class is designed for dense matrices
    """
    def __init__(self, Y, initialize_lists=False):
        self.Y = Y
        self.M, self.R = Y.shape

    def initialize_sorted_lists(self):
        """
        Initializes the sorted lists
        """
        self.sorted_lists = self.Y.argsort(0)

    def score_item(self, x_u, indice):
        return (dot(self.Y[indice], x_u), indice)

    def update_top_list(self, top_list, new_scored_item, K, n_scored):
        """
        Updates the top-K list with a new scored item
        """
        if n_scored < K:
            top_list.append( new_scored_item )
            top_list.sort()
        elif new_scored_item[0] > top_list[0][0]:
            # case that the new item is better than the worst in the list
            top_list[0] = new_scored_item  # replace the worst with the current
            top_list.sort()  # resort the list
            # python should be able to cope with partially ordered lists
            # so this is expected to be fast
        # returns nothing, processes the list

    def get_top_K_naive(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K for a given query by naively scoring all the items
        """
        top_list = []
        n_items_scored = 0
        for indice in range(self.M):
            new_scored_item = self.score_item(x_u, indice)
            self.update_top_list(top_list, new_scored_item, K,
                n_items_scored)
            n_items_scored += 1
        if count_calculations:
            return top_list, n_items_scored
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
        top_list = []
        n_items_scored = 0
        neg_elements_query = set([i for i, el in enumerate(x_u) if el < 0])
        non_zero_elements_query = [i for i, el in enumerate(x_u) if el != 0]
        scored = set([])
        upper_bound = 1
        lower_bound = 0
        depth = 0
        while upper_bound > lower_bound:
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
                    if n_items_scored < K or lower_bound < new_scored_item[0]:
                            self.update_top_list(top_list, new_scored_item, K,
                                n_items_scored)
                    n_items_scored += 1
                    scored.add(item)
            # update lower bound
            if n_items_scored >= K:
                lower_bound = top_list[0][0]
            depth += 1
        if count_calculations:
            return top_list, n_items_scored
        else:
            return top_list

    def get_top_K_threshold_enhanced(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K using the modified threshold algorithm
        """
        top_list = []
        n_items_scored = 0
        # initiate list with rules
        # contains tuples with
        # (paritial score, xi, r, position_sorted_list, decr/incr sorted)
        query_info_list = [(calculate_partial_score(r, 0, xi, self.Y, self.sorted_lists),
                        xi,
                        r,
                        0 if xi < 0 else -1,
                        1 if xi < 0 else -1)\
                for r, xi in enumerate(x_u) if xi != 0]
        query_info_list.sort(reverse=True)
        scored = set([])
        #  we start with the upper bound which we update iteratively
        upper_bound = sum([x[0] for x in query_info_list])
        lower_bound = -1e100
        while upper_bound > lower_bound:
            partial_score, xi, r, pos, pos_action = query_info_list[0]
            # get item
            item = self.sorted_lists[pos, r]
            # score item
            if item not in scored:
                new_scored_item = self.score_item(x_u, item)
                if n_items_scored < K or lower_bound < new_scored_item[0]:
                    self.update_top_list(top_list, new_scored_item, K,
                        n_items_scored)
                n_items_scored += 1
                scored.add(item)
            # update lower bound
            if n_items_scored >= K:
                lower_bound = top_list[0][0]
            # update position for this list
            pos += pos_action
            # update the upper bound
            upper_bound -= partial_score  # remove previous partial score
            partial_score = xi * self.Y[item, r]  # get new partial score
            upper_bound += partial_score
            # update the rule list
            query_info_list[0] = (partial_score, xi, r, pos, pos_action)
            cure_partial_ordered_list(query_info_list)  # keep ordering
        if count_calculations:
            return top_list, n_items_scored
        else:
            return top_list



if __name__ == '__main__':

    import numpy as np

    R = 25

    W = np.random.rand(100000, R)**2

    inferer = TopKInference(W)

    x = np.random.randn(R)**2
    #x[3] = 100

    top_5_list_naive, n_scored_naive = inferer.get_top_K_naive(x, 5, True)

    inferer.initialize_sorted_lists()

    top_5_list_threshold, n_scored_threshold = inferer.get_top_K_threshold(x, 10, True)

    top_5_list_threshold_enh, n_scored_threshold_enh = inferer.get_top_K_threshold_enhanced(x, 10, True)

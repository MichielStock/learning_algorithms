"""
Created on Tue Jun 9 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of algorithms for the efficient inference of the top-K of
seperable linear models
"""

from numpy import dot

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

    def update_top_list(self, top_list, new_scored_item, K, n_scored,
            worst_in_list):
        """
        Updates the top-K list with a new scored item
        """
        if n_scored < K:
            top_list.append( new_scored_item )
            top_list.sort()
        elif new_scored_item[0] > worst_in_list:
            # case that the new item is better than the worst in the list
            top_list[0] = new_scored_item  # replace the worst with the current
            top_list.sort()  # resort the list
            worst_in_list = top_list[0][0]
            # python should be able to cope with partially ordered lists
            # so this is expected to be fast
        # returns nothing, processes the list
        return worst_in_list

    def get_top_K_naive(self, x_u, K=1, count_calculations=False):
        """
        Returns top-K for a given query by naively scoring all the items
        """
        top_list = []
        n_items_scored = 0
        worst_in_list = -1e100
        for indice in range(self.M):
            new_scored_item = self.score_item(x_u, indice)
            worst_in_list = self.update_top_list(top_list, new_scored_item, K,
                n_items_scored, worst_in_list)
            n_items_scored += 1
        if count_calculations:
            return top_list, count_calculations
        else:
            return top_list


if __name__ == '__main__':

    import numpy as np

    R = 100

    W = np.random.randn(10000, R)

    inferer = TopKInference(W)

    x = np.random.rand(R)

    top_5_list = inferer.get_top_K_naive(x, 5)

"""
Created on Sun Jun 14 2015
Last update: Fri Jun 19 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of heaps using numpy and jit
Goal is to devellop algorithms compatible with jit
"""

import numpy as np
from numba import jit

# Funtions for max heaps
# ----------------------

@jit
def max_heapify_from_i(values, indices, i):
    # cures a heap from position i onwards, for max heaps
    heaplength = len(values)
    left = i*2 + 1
    right = i*2 + 2
    largest = i
    if left < heaplength and values[left] > values[largest]:
        largest = left
    if right < heaplength and values[right] > values[largest]:
        largest = right
    if largest != i:
        values[[i, largest]] = values[[largest, i]]
        indices[[i, largest]] = indices[[largest, i]]
        max_heapify_from_i(values, indices, largest)

@jit
def max_heapify(values, indices):
    # turns values and associated indices into a max heap
    heaplength = len(values)
    for i in range(heaplength):
        max_heapify_from_i(values, indices, heaplength-i-1)

# Functions for min heaps
# -----------------------

@jit
def min_heapify_from_i(values, indices, i):
    # cures a heap from position i onwards, for min heaps
    heaplength = len(values)
    left = i*2 + 1
    right = i*2 + 2
    smallest = i
    if left < heaplength and values[left] < values[smallest]:
        smallest = left
    if right < heaplength and values[right] < values[smallest]:
        smallest = right
    if smallest != i:
        values[[i, smallest]] = values[[smallest, i]]
        indices[[i, smallest]] = indices[[smallest, i]]
        min_heapify_from_i(values, indices, smallest)

@jit
def min_heapify(values, indices):
    # turns values and associated indices into a min heap
    heaplength = len(values)
    for i in range(heaplength):
        min_heapify_from_i(values, indices, heaplength-i-1)

# Some nice classes
# -----------------

class MaxHeap():
    # max heap with FIXED number of values and indices

    def __init__(self, values, indices, heapify=True):
        self._values = values
        self._indices = indices
        if heapify:
            self.heapify()

    def heapify(self):
        max_heapify(self._values, self._indices)

    def peek(self):
        # return top element
        return (self._values[0], self._indices[0])

    def heapreplace(self, (val, ind)):
        # replaces the best with a new value while keeping the heap property
        self._values[0] = val
        self._indices[0] = ind
        max_heapify_from_i(self._values, self._indices, 0)

    def heapupdate(self, (val, ind)):
        # adds an element if it outperforms the top
        if val < self._values[0]:
            self.heapreplace((val, ind))

class MinHeap(MaxHeap):
    # min heap with FIXED number of values and indices

    def heapify(self):
        min_heapify(self._values, self._indices)

    def heapreplace(self, (val, ind)):
        # replaces the best with a new value while keeping the heap property
        self._values[0] = val
        self._indices[0] = ind
        min_heapify_from_i(self._values, self._indices, 0)

    def heapupdate(self, (val, ind)):
        # adds an element if it outperforms the top
        if val > self._values[0]:
            self.heapreplace((val, ind))

def validate_heap(values, maxheap=True):
    # check if the heap propperty is fulfilled
    heaplen = len(values)
    for i in range(heaplen):
        for j in [1,2]:
            if 2*i + j < heaplen:
                if maxheap:
                    if values[i] > values[2*i + j]:
                        print 'ok'
                    else:
                        print 'not ok'
                elif (not maxheap):
                    if values[i] < values[2*i + j]:
                        print 'ok'
                    else:
                        print 'not ok'



if __name__ == '__main__':

    a = np.random.randn(10000)
    b = np.arange(10000)

    max_heapify(a,b)
    validate_heap(a)

    min_heapify(a,b)
    validate_heap(a, maxheap=False)

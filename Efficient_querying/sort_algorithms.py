"""
Created on Sun Dec 28 2014
Last update: Sun Dec 28 2014

@author: Michiel Stock
michielfmstock@gmail.com

Some very basic sorting functions and matrix multiplications
written in pure Python code to assess performance
"""

import random as rd

def issorted(arr, left, right):
    sorted = True
    for i in range(left, right-1):
        if arr[i] > arr[i+1]:
            sorted = False
            break
    return sorted


def partition(array, left, right, pivotIndex):
    '''
    Function needed as a subroutine for quicksort
    '''
    pivotValue = array[pivotIndex]
    array[pivotIndex] = array[right]
    array[right] = pivotValue
    storeIndex = left
    for i in xrange(left, right):
        if array[i] <= pivotValue:
            val = array[i]
            array[i] = array[storeIndex]
            array[storeIndex] = val
            storeIndex += 1
    val = array[right]
    array[right] = array[storeIndex]
    array[storeIndex] = val
    return storeIndex
        
def quicksort(array, left = 0, right = -1):
    '''
    problem: crashes on sorted list, think to solve
    '''
    if len(array) > 1:
	    while right < 0:
	        right += len(array)
	    if left < right:
	        pivotIndex = rd.choice(range(left, right + 1))	        
	        pivotNewIndex = partition(array, left, right, pivotIndex)
	        quicksort(array, left, pivotNewIndex - 1)
	        quicksort(array, pivotNewIndex + 1, right)
    return array
    
def safequicksort(array):
    if not issorted(array, 0, len(array)-1):
        return quicksort(array)
    else:
        return array
    
def argsort(seq):
    """
    performs an argsort on a list, based on own implementation of quicksort 
    """
    return [tup[1] for tup in quicksort([(seq[i], i) for i in range(len(seq))])]

def poor_mans_dot(u,v):
    result = 0.0
    for i in range(len(u)):
        result += u[i]*v[i]
    return result

def poor_mans_transpose(A):
    return map(list, zip(*A))
    
def poor_mans_matrix_mult(A, B, Btransp = True):
    '''
    multiplies a n * k matrix A with a k * m matrix B
    '''
    nrA = len(A)
    ncA = len(A[0])
    nrB = len(B)
    ncB = len(B[0])
    if Btransp:
        M = [[0.0 for j in xrange(nrB)] for i in xrange(nrA)]
        for i in xrange(nrA):
            for j in xrange(nrB):
                for k in xrange(ncA):
                    M[i][j] += A[i][k]*B[j][k]
        return M
    else:
        M = [[0.0 for j in xrange(ncB)] for i in xrange(nrA)]
        for i in xrange(nrA):
            for j in xrange(ncB):
                for k in xrange(ncA):
                    M[i][j] += A[i][k]*B[k][j]
        return M
        
def poor_mans_rowSum(X):
    n = len(X)
    m = len(X[0])
    S = []
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += X[i][j]
        S.append(s)
    return S
    
def poor_mans_sum(x):
    s = 0.0
    for xi in x:
        s += xi
    return s

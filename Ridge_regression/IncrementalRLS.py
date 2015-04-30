"""
Created on Thu Apr 30 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of an incremental RLS
"""
import numpy as np

class IncrementalRLS():
    '''
    Ridge regression which can be trained in a set of mini-batches
    '''
    def __init__(self, Yinit, Xinit, l=1.0):
        n, p = Xinit.shape
        self._p = p
        self._lambda = l
        self._XTY = np.dot(Xinit.T, Yinit)
        self._Ainv = np.linalg.inv(np.dot(Xinit.T, Xinit) +\
                np.eye(p) * self._lambda)
        self._W = np.dot(self._Ainv, self._XTY)

    def predict(self, Xpred):
        return np.dot(Xpred, self._W)

    def update(self, Ynew, Xnew):
        n = Xnew.shape[0]
        self._XTY += np.dot(Xnew.T, Ynew)
        Xnew_x_Ainv = np.dot(Xnew, self._Ainv)
        C = np.linalg.inv(np.eye(n) + Xnew_x_Ainv.dot(Xnew.T))
        self._Ainv -= np.dot(Xnew_x_Ainv.T, C).dot(Xnew_x_Ainv)
        self._W = np.dot(self._Ainv, self._XTY)

if __name__ == '__main__':
    Ybig = np.random.randn(1000, 10)
    Xbig = np.random.randn(1000, 100)

    Xtest = np.random.randn(50, 100)

    # model in one step

    model_complete = IncrementalRLS(Ybig, Xbig)
    Ypred1 = model_complete.predict(Xtest)

    # model in two steps

    model_stepwise = IncrementalRLS(Ybig[:500], Xbig[:500])
    model_stepwise.update(Ybig[500:], Xbig[500:])
    Ypred2 = model_stepwise.predict(Xtest)

    # are they the same

    print np.allclose(Ypred1, Ypred2)

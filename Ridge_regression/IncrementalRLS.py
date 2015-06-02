"""
Created on Thu Apr 30 2015
Last update: Tue Jun 02 2015

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
        #self._XTY = np.dot(Xinit.T, Yinit)
        self._XTY = (Yinit.T.dot(Xinit)).T
        self._A = np.dot(Xinit.T, Xinit) + np.eye(p) * self._lambda
        self._Ainv = np.linalg.inv(self._A)
        self._W = np.dot(self._Ainv, self._XTY)

    def predict(self, Xpred):
        return np.dot(Xpred, self._W)

    def update(self, Ynew, Xnew):
        n, p = Xnew.shape
        #self._XTY += np.dot(Xnew.T, Ynew)
        self._XTY += (Ynew.T.dot(Xnew)).T
        Xnew_x_Ainv = np.dot(Xnew, self._Ainv)
        C = np.linalg.inv(np.eye(n) + Xnew_x_Ainv.dot(Xnew.T))
        self._Ainv -= np.dot(Xnew_x_Ainv.T, C).dot(Xnew_x_Ainv)
        self._W = np.dot(self._Ainv, self._XTY)

class IncrementalTwoStepRLS(IncrementalRLS):
    '''
    Two-step ridge regression which can be trained in a set of mini-batches
    currently only for fixed tasks
    '''
    def __init__(self, Yinit, Xu_init, Xv_init, l1=1.0, l2=1.0):
        n, p = Xu_init.shape
        m, q = Xv_init.shape
        self._p = p
        self._lambda = (l1, l2)
        #self._XTY = np.dot(Xinit.T, Yinit)
        self._XTY = (Yinit.T.dot(Xu_init)).T
        A_tasks = np.linalg.inv(Xv_init.T.dot(Xv_init) + l2*np.eye(q))
        self._XTY = self._XTY.dot(Xv_init).dot(A_tasks).dot(Xv_init.T)
        self._A = np.dot(Xu_init.T, Xu_init) + np.eye(p) * self._lambda
        self._Ainv = np.linalg.inv(self._A)
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

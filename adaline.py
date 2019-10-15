import numpy as np
from random import randint
from sklearn.preprocessing import LabelEncoder


class AdaLine(object):
    def __init__(self, eta=0.01, T=1000):
        self.eta = eta
        self.T = T
        self.w0 = 0
        self.w = None
        
    def fit(self, X, y):
        lenc = LabelEncoder()
        X, y = np.array(X), np.array(y)
        n = len(X[0])
        self.w0 = 0
        self.w = np.zeros(shape=n)
        for t in range(self.T):
            i = randint(0,n)
            hx = self.hcalc(X[i])
            self.w0 += self.eta * (y[i] - hx)
            self.w += self.eta * (y[i] - hx) * X[i]
    
    def hcalc(self, X):
        value = np.dot(self.w, X) + self.w0
        return 1 if value >= 0 else -1
    
    def predict(self, X):
        X = np.array(X)
        n = len(X)
        return [self.hcalc(X[i]) for i in range(n)]

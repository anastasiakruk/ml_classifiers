import numpy as np
import os.path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataSets(object):
    def __init__(self):
        self.data = []
        self.dirname = "./data"
        self.initData()
    
    def initData(self):
        func = [self.breastCancerWisconsinData,
                self.wdbcData,
                self.wpbcData,
                self.ionosphereData,
                self.agaricusLepiotaData]
        for f in func:
            self.data.append(f())
    
    def breastCancerWisconsinData(self):
        path = os.path.join(self.dirname, 
                            "breast-cancer-wisconsin.data")
        df = pd.read_csv(path)
        df = self.delScips(df)
        X, y = df[df.columns[1:-1]], df[df.columns[-1]]
        def yProc(x):
            if x == 2:
                return 1
            if x == 4:
                return -1
        
        X, y = self.dataProc(X), list(map(yProc, y))
        return path, X, y

    def wdbcData(self):
        path = os.path.join(self.dirname, "wdbc.data")
        df = pd.read_csv(path)
        X, y = df[df.columns[2:]], df[df.columns[1]]
        def yProc(x):
            if x == 'B':
                return 1
            if x == 'M':
                return -1
        
        X, y = self.dataProc(X), list(map(yProc, y))
        return path, X, y

    def wpbcData(self):
        path = os.path.join(self.dirname, "wpbc.data")
        df = pd.read_csv(path)
        df = self.delScips(df)
        X, y = df[df.columns[2:]], df[df.columns[1]]
        def yProc(x):
            if x == 'N':
                return 1
            if x == 'R':
                return -1

        X, y = self.dataProc(X), list(map(yProc, y))
        return path, X, y

    def ionosphereData(self):
        path = os.path.join(self.dirname, "ionosphere.data")
        df = pd.read_csv(path)
        X, y = df[df.columns[0:-1]], df[df.columns[-1]]
        def yProc(x):
            if x == "b":
                return -1
            if x == "g":
                return 1
        
        X, y = self.dataProc(X), list(map(yProc, y))
        return path, X, y

    def agaricusLepiotaData(self):
        path = os.path.join(self.dirname, "agaricus-lepiota.data")
        df = pd.read_csv(path)
        df = self.delScips(df)
        X, y = df[df.columns[1:]], df[df.columns[0]]
        def yProc(x):
            if x == 'e':
                return 1
            if x == 'p':
                return -1

        X, y = self.dataProc(X, le=True), list(map(yProc, y))
        return path, X, y

    def dataProc(self, X0, le=False):
        X = X0.copy()
        if le:
            lenc = LabelEncoder()
            X = X.apply(lenc.fit_transform)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def delScips(self, df):
        df = np.array(df)
        ni, nj = len(df), len(df[0])
        for i in range(ni):
            for j in range(nj):
                if df[i][j] == "?":
                    df[i][j] = None
        df = pd.DataFrame(df).dropna()
        return df
    
    def __iter__(self):
        for path, X, y in self.data:
            yield path, X, y

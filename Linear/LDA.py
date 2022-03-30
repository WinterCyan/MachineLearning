import numpy as np
import pandas as pd

class LDA:
    def __init__(self):
        self.w = None
        self.u0 = None
        self.u1 = None

    def fit(self,X,y):
        pos = y==1
        neg = y==0
        X0 = X[pos]
        X1 = X[neg]
        u0 = np.mean(X0, axis=0, keepdims=True)
        u1 = np.mean(X1, axis=0, keepdims=True)
        Sw = np.dot((X0-u0).T, X0-u0) + np.dot((X1-u1).T, X1-u1)
        w = np.dot(np.linalg.inv(Sw), (u0-u1).T).reshape(1,-1)  # calculate directly, Sw^-1(u0-u1)
        self.w = w
        self.u0 = u0
        self.u1 = u1

    def predict(self,X):
        proj = np.dot(X, self.w.T)
        new_u0 = np.dot(self.w, self.u0.T)
        new_u1 = np.dot(self.w, self.u1.T)
        return (np.abs(proj-new_u1)<np.abs(proj-new_u0)).astype(int)


if __name__ == '__main__':
    data = pd.read_csv("../data/watermelon3_0_Ch.csv").values

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    lda = LDA()
    lda.fit(X, y)
    print(lda.predict(X))
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class Solver:
    def __init__(self):
        pass

    def predict(self,x):
        pass

    def fit(self,X,y):
        pass


class SVM(Solver):
    def __init__(self, c:float=1., kkt_thr:float=1e-3, train_iters:int=1e4, kernel_type:str='linear', gamma_rbf:float=1.):
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError('unkown kernel type...')
        super(SVM, self).__init__()
        self.c = float(c)
        self.kkt_thr = kkt_thr
        self.train_iters = train_iters
        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

        self.sv = np.array([])
        self.sv_label = np.array([])
        self.b = 0.0
        self.alphas = np.array([])

    def predict(self,x):
        w = self.sv_label*self.alphas
        x = self.kernel(self.sv, x)
        scores = np.matmul(w,x) + self.b
        pred = np.sign(scores)
        return pred, scores

    def fit(self,X,y):
        n, d = X.shape[0], X.shape[1]
        self.b = 0
        self.alphas = np.zeros([n])
        self.sv, self.sv_label = X,y
        non_kkts = np.arange(n)
        err_cache = self.predict(X)[1] - y

        iter = 0
        while iter < self.train_iters:
            i2, non_kkts = self.i2_heuristic(non_kkts)
            if i2 == -1:
                break
            i1 = self.i1_heuristic(i2, err_cache)
            if i1 == i2:
                continue

            # get corresponding sample
            x1, y1, alpha1 = self.sv[i1,:], self.sv_label[i1], self.alphas[i1]
            x2, y2, alpha2 = self.sv[i2,:], self.sv_label[i2], self.alphas[i2]

            # calculate L & H
            low, high = self.cal_bound(alpha1, alpha2, y1, y2)
            if low == high:
                continue
            # calculate \eta
            eta = self.cal_eta(x1, x2)
            if eta == 0:
                continue

            # calculate predictions and err
            _, score1 = self.predict(x1)
            _, score2 = self.predict(x2)
            err1 = score1 - y1
            err2 = score2 - y2

            # update alpha2
            alpha2_new = alpha2 + y2 * (err1-err2)/eta
            alpha2_new = np.minimum(alpha2_new, high)
            alpha2_new = np.maximum(alpha2_new, low)

            alpha1_new = alpha1 + y1*y2*(alpha2-alpha2_new)

            # update alpha1
            self.cal_b(alpha1_new, alpha2_new, err1, err2, i1, i2)

            # update err cache
            self.alphas[i1] = alpha1_new
            self.alphas[i2] = alpha2_new
            err_cache[i1] = self.predict(x1)[1] - y1
            err_cache[i2] = self.predict(x2)[1] - y2

            iter += 1

        sv_idx = (self.alphas != 0)
        self.sv_label = self.sv_label[sv_idx]
        self.sv = self.sv[sv_idx,:]
        self.alphas = self.alphas[sv_idx]

    def i1_heuristic(self, i_2, err_cache):
        err2 = err_cache[i_2]
        # get all 0<alpha<c
        non_bounded = np.argwhere((self.alphas>0) & (self.alphas<self.c)).reshape((1,-1))[0]
        if non_bounded.shape[0] > 0:
            # get max |E1-E2|
            if err2 >= 0:
                i_1 = non_bounded[np.argmin(err_cache[non_bounded])]
            else:
                i_1 = non_bounded[np.argmax(err_cache[non_bounded])]
        else:
            i_1 = np.argmax(np.abs(err_cache-err2))
        return i_1

    def i2_heuristic(self, non_kkts):
        """
        choose an alpha to update
        :param non_kkts: left alpha indexes
        :return: selected alpha
        """
        i_2 = -1
        # select first
        for idx in non_kkts:
            # delete once checked; when non-kkt: delete&select and break, when kkt: delete and continue
            non_kkts = np.delete(non_kkts, np.argwhere(non_kkts==idx))
            # find first violating alpha
            if not self.valid_kkt(idx):
                i_2 = idx
                break

        if i_2 == -1:
            # all satisfies
            idxs = np.arange(self.alphas.shape[0])
            non_kkts = idxs[~(self.valid_kkt(idxs))]
            if non_kkts.shape[0]>0:
                np.random.shuffle(non_kkts)
                i_2 = non_kkts[0]
                non_kkts = non_kkts[1:-1]

        return i_2, non_kkts

    def valid_kkt(self,idx):
        alpha = self.alphas[idx]
        _, score = self.predict(self.sv[idx, :])
        y = self.sv_label[idx]
        # r = yi * f(xi) - 1
        r = y * score - 1
        # condition: when alpha<C, must be; yi*f(xi)-1+eps >= 0
        cond1 = (alpha<self.c) & (r+self.kkt_thr<0)     # if alpha > c, cond1 = false
        # condition: when alpha>0, must be: yi*f(xi)==1-eps
        cond2 = (alpha>0) & (r>self.kkt_thr)            # if alpha == 0, cond2 = false
        return ~(cond1|cond2)

    def cal_bound(self, alpha1, alpha2, y1, y2):
        if y1 == y2:
            low = np.max([0, alpha1+alpha2-self.c])
            high = np.min([self.c, alpha1+alpha2])
        else:
            low = np.max([0, alpha2-alpha1])
            high = np.min([self.c, self.c+alpha2-alpha1])

        return low, high

    def cal_eta(self, x1, x2):
        return self.kernel(x1,x1)+self.kernel(x2,x2)-2.0*self.kernel(x1,x2)


    def cal_b(self,alpha1_new,alpha2_new,err1,err2,i1,i2):
        x1,x2 = self.sv[i1], self.sv[i2]
        b1 = self.b-(err1 + self.sv_label[i1]*(alpha1_new-self.alphas[i1])*self.kernel(x1,x1)+self.sv_label[i2]*(alpha2_new-self.alphas[i2])*self.kernel(x1,x2))
        b2 = self.b-(err2 + self.sv_label[i1]*(alpha1_new-self.alphas[i1])*self.kernel(x1,x2)+self.sv_label[i2]*(alpha2_new-self.alphas[i2])*self.kernel(x2,x2))
        if 0<alpha1_new<self.c:
            self.b = b1
        elif 0<alpha2_new<self.c:
            self.b = b2
        else:
            self.b = np.mean([b1,b2])

    @staticmethod
    def linear_kernel(u,v):
        return np.dot(u,v.T)

    def rbf_kernel(self,u,v):
        if np.ndim(v) == 1:
            v = v[np.newaxis,:]
        if np.ndim(u) == 1:
            u = u[np.newaxis,:]
        dist_squared = np.linalg.norm(u[:,:,np.newaxis]-v.T[np.newaxis,:,:], axis=1)**2
        dist_squared = np.squeeze(dist_squared)
        return np.exp(-self.gamma_rbf*dist_squared)

class MultiClassifier:
    def __init__(self, solver:Solver, num_class:int, **kwargs):
        # how to use **kwargs
        self._binary_classifiers = [solver(**kwargs) for i in range(num_class)]
        self._num_classes = num_class

    def predict(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self._num_classes))
        for idx in range(self._num_classes):
            scores[:,idx] = self._binary_classifiers[idx].predict(X)[1]
        pred = np.argmax(scores, axis=1)
        return pred

    def fit(self,X,y):
        for idx in range(self._num_classes):
            # labels to {+1,-1}
            # y: 0 0 0 ~ 1 1 1 ~ 2 2 2
            # idx=0, 1 1 1 ~ -1 -1 -1 ~ -1 -1 -1
            # idx=1, -1 -1 -1 ~ 1 1 1 ~ -1 -1 -1
            y_cvt = 1.0*(y == idx) - 1.0*(y != idx)
            self._binary_classifiers[idx].fit(X,y_cvt)



if __name__ == '__main__':
    iris_data = load_iris()
    # iris_data.keys: (['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target,shuffle=True, test_size=0.3,stratify=iris_data.target)
    print(f'Dataset split summary:')
    print(f'Training set size: {X_train.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')
    iris_df = pd.DataFrame(X_train, columns=iris_data.feature_names)
    # iris_df.columns: (['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], dtype='object')
    iris_df['species'] = iris_data.target_names[y_train.reshape(-1,1)]
    # pairplot, 属性两两之间的关系
    sns.pairplot(iris_df, hue=iris_df.columns[-1])
    # plt.show()

    solver = MultiClassifier(solver=SVM, num_class=len(iris_data.target_names), c=1.0, kkt_thr=1e-3, train_iters=1e3, kernel_type='rbf', gamma_rbf=1.0)
    solver.fit(X_train, y_train)
    y_pred = solver.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=iris_data.target_names))

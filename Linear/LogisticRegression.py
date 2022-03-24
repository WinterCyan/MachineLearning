import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

class LR:
    def __init__(self, dim, lr, train_iters):
        self.lr = lr
        self.train_iters = train_iters
        self.num_feat = dim
        self.beta = np.array((self.num_feat+1,1))

    def fit(self,X,y):
        self.__init_weight()
        X = np.c_[X, np.ones((X.shape[0],1))]
        for i in range(self.train_iters):
            grad = self.__cal_gradient(X,y)
            self.beta -= self.lr * grad
            if i%10 == 0:
                print(f'{i}th iter, cost is {self.__cal_cost_J(X,y)}...')

    def predict(self,X):
        X = np.c_[X, np.ones((X.shape[0],1))]
        p1 = sigmoid(np.dot(X,self.beta))
        p1[p1>=0.5] = 1
        p1[p1<0.5] = 0
        return p1

    def __cal_cost_J(self,X,y):
        beta = self.beta.reshape(-1,1)
        y = y.reshape(-1,1)
        L_beta = -y*np.dot(X,beta) + np.log(1+np.exp(np.dot(X,beta)))
        return L_beta

    def __cal_gradient(self,X,y):
        """
        :param X: shape: (N,D)
        :param y: shape: (N,)
        :return:
        """
        beta = self.beta.reshape(-1,1)
        y = y.reshape(-1,1)
        p1 = sigmoid(np.dot(X,beta))
        grad = (-X * (y-p1)).sum(0)
        return grad.reshape(-1,1)

    def __init_weight(self):
        self.beta = np.random.random((self.num_feat+1,1))

if __name__ == '__main__':
    data = pd.read_csv('watermelon3_0_Ch.csv').values
    # [17,10]
    print(data)

    good_idx = data[:,-1] == '是'
    bad_idx = data[:,-1] == '否'

    X = data[:,7:9].astype(float)
    y = data[:,9]
    y[y=='是'] = 1
    y[y=='否'] = 0
    y = y.astype(int)

    plt.scatter(data[:,7][good_idx], data[:,8][good_idx], c='k', marker='o')
    plt.scatter(data[:,7][bad_idx], data[:,8][bad_idx], c='r', marker='x')

    # print(type(X))
    # print(type(y))

    model = LR(dim=X.shape[1], lr=1.2, train_iters=100)
    model.fit(X,y)

    w1, w2, b = model.beta
    xx = np.linspace(0,1)
    yy = -(w1*xx+b)/w2
    ax1 = plt.plot(xx,yy)

    plt.show()

    pred = model.predict(X)
    print(pred)


#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn import linear_model
#
#
# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     return s
#
#
# def J_cost(X, y, beta):
#     '''
#     :param X:  sample array, shape(n_samples, n_features)
#     :param y: array-like, shape (n_samples,)
#     :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
#     :return: the result of formula 3.27
#     '''
#     X_hat = np.c_[X, np.ones((X.shape[0], 1))]
#     beta = beta.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#
#     Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))
#
#     return Lbeta.sum()
#
#
# def gradient(X, y, beta):
#     '''
#     compute the first derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.30
#     ----------------------------------
#     :param X: sample array, shape(n_samples, n_features)
#     :param y: array-like, shape (n_samples,)
#     :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
#     :return:
#     '''
#     X_hat = np.c_[X, np.ones((X.shape[0], 1))]
#     beta = beta.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#     p1 = sigmoid(np.dot(X_hat, beta))
#
#     gra = (-X_hat * (y - p1)).sum(0)
#
#     return gra.reshape(-1, 1)
#
#
# def hessian(X, y, beta):
#     '''
#     compute the second derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.31
#     ----------------------------------
#     :param X: sample array, shape(n_samples, n_features)
#     :param y: array-like, shape (n_samples,)
#     :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
#     :return:
#     '''
#     X_hat = np.c_[X, np.ones((X.shape[0], 1))]
#     beta = beta.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#
#     p1 = sigmoid(np.dot(X_hat, beta))
#
#     m, n = X.shape
#     P = np.eye(m) * p1 * (1 - p1)
#
#     assert P.shape[0] == P.shape[1]
#     return np.dot(np.dot(X_hat.T, P), X_hat)
#
#
# def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
#     '''
#     update parameters with gradient descent method
#     --------------------------------------------
#     :param beta:
#     :param grad:
#     :param learning_rate:
#     :return:
#     '''
#     for i in range(num_iterations):
#
#         grad = gradient(X, y, beta)
#         beta = beta - learning_rate * grad
#
#         if (i % 10 == 0) & print_cost:
#             print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
#
#     return beta
#
#
# def update_parameters_newton(X, y, beta, num_iterations, print_cost):
#     '''
#     update parameters with Newton method
#     :param beta:
#     :param grad:
#     :param hess:
#     :return:
#     '''
#
#     for i in range(num_iterations):
#
#         grad = gradient(X, y, beta)
#         hess = hessian(X, y, beta)
#         beta = beta - np.dot(np.linalg.inv(hess), grad)
#
#         if (i % 10 == 0) & print_cost:
#             print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
#     return beta
#
#
# def initialize_beta(n):
#     beta = np.random.randn(n + 1, 1) * 0.5 + 1
#     return beta
#
#
# def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
#     '''
#     :param X:
#     :param y:~
#     :param num_iterations:
#     :param learning_rate:
#     :param print_cost:
#     :param method: str 'gradDesc' or 'Newton'
#     :return:
#     '''
#     m, n = X.shape
#     beta = initialize_beta(n)
#
#     if method == 'gradDesc':
#         return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
#     elif method == 'Newton':
#         return update_parameters_newton(X, y, beta, num_iterations, print_cost)
#     else:
#         raise ValueError('Unknown solver %s' % method)
#
#
# def predict(X, beta):
#     X_hat = np.c_[X, np.ones((X.shape[0], 1))]
#     p1 = sigmoid(np.dot(X_hat, beta))
#
#     p1[p1 >= 0.5] = 1
#     p1[p1 < 0.5] = 0
#
#     return p1
#
#
# if __name__ == '__main__':
#     #
#     data = pd.read_csv("watermelon3_0_Ch.csv").values
#
#     is_good = data[:, 9] == '是'
#     is_bad = data[:, 9] == '否'
#
#     X = data[:, 7:9].astype(float)
#     y = data[:, 9]
#
#     y[y == '是'] = 1
#     y[y == '否'] = 0
#     y = y.astype(int)
#
#     plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')
#     plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')
#
#     plt.xlabel('密度')
#     plt.ylabel('含糖量')
#
#     # 可视化模型结果
#     beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
#     w1, w2, intercept = beta
#     x1 = np.linspace(0, 1)
#     y1 = -(w1 * x1 + intercept) / w2
#
#     ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')
#
#     lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
#     lr.fit(X, y)
#
#     lr_beta = np.c_[lr.coef_, lr.intercept_]
#     print(J_cost(X, y, lr_beta))
#
#     # 可视化sklearn LogisticRegression 模型结果
#     w1_sk, w2_sk = lr.coef_[0, :]
#
#     x2 = np.linspace(0, 1)
#     y2 = -(w1_sk * x2 + lr.intercept_) / w2
#
#     ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')
#
#     plt.legend(loc='upper right')
#     plt.show()
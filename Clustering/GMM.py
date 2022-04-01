import numpy as np
from scipy.stats import multivariate_normal
from numpy import genfromtxt
from matplotlib import pyplot as plt

class GMM:
    def __init__(self, n_feature, n_class, eps=1e-6):
        """
        :param n_feature: D
        :param n_class: K
        """
        self.K = n_class
        self.D = n_feature
        self.eps = eps

        self.gaussian_means = None  # K*D
        self.gaussian_vars = None  # K*D*D
        self.gaussian_weights = None  # K
        self.gamma = None  # shape: N*K

        self.X = None  # shape: N*D
        self.N = None

    def __init_gaussian(self,X):
        self.X = X
        self.N = X.shape[0]
        np.random.shuffle(X)
        means = []
        covs = []
        div_size = int(np.floor(self.N/self.K))
        X_splits = np.array([X[i:i+div_size] for i in range(0, self.N, div_size)])
        self.gaussian_means = np.zeros((self.K, self.D))
        self.gaussian_vars = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            means.append(np.mean(X_splits[k], axis=0))
            covs.append(np.cov(X_splits[k].T))
        self.gaussian_means = np.array(means)
        self.gaussian_vars = np.array(covs)

        self.gaussian_weights = np.ones(self.K)/self.K
        self.gamma = np.zeros((self.N, self.K))
        print(self.gaussian_means)
        print(self.gaussian_vars)
        print(self.gaussian_weights)

    def clustering(self,X):
        assert X.shape[1] == self.D
        N = X.shape[0]
        res = np.zeros(N).astype(int)
        ma_dist = np.zeros(self.K).astype(float)
        conv_invs = [np.linalg.inv(var) for var in self.gaussian_vars]
        for i in range(N):
            x = X[i,:]
            for k in range(self.K):
                ma_dist[k] = np.dot(np.dot((x-self.gaussian_means[k]).T, conv_invs[k]), x-self.gaussian_means[k])
            res[i] = np.argmin(ma_dist)
        return res


    def fit(self,X,train_iters):
        """
        :param X: shape: N*D
        :param train_iters: iters to train
        :return:
        """
        prev_loglike = -99999
        self.__init_gaussian(X)
        for iter in range(train_iters):
            self.E_step()
            self.M_step()
            loglike = self.__logLike(self.X, self.gaussian_weights, self.gaussian_means, self.gaussian_vars)
            if abs(prev_loglike-loglike)<self.eps:
                break
            prev_loglike = loglike
            print(f"iter:{iter}, loss:{prev_loglike}")

    def E_step(self):
        # gamm ji: sample xj -> gaussian gi, shape N*K
        # p(xj | gi): shape 1*1
        # gaussian: mu: [D,1], sigma: [D,1]; sample: [N,D]
        for i in range(self.N):
            x = self.X[i,:]
            p_sum = 0
            # calculate all p for model k
            for k in range(self.K):
                p = multivariate_normal.pdf(x, self.gaussian_means[k,:], self.gaussian_vars[k,:,:])
                p_sum += self.gaussian_weights[k] * p
            # normalize p as sum as 1
            for k in range(self.K):
                self.gamma[i, k] = self.gaussian_weights[k] * multivariate_normal.pdf(x, self.gaussian_means[k,:], self.gaussian_vars[k,:,:])/p_sum


    def M_step(self):
        new_mu = np.zeros((self.K, self.D)).astype(float)
        new_sigma = np.zeros((self.K, self.D, self.D)).astype(float)
        new_weights = np.zeros(self.K).astype(float)
        for k in range(self.K):
            # get sum of post-p
            post_p_sum = np.sum(self.gamma, axis=0)[k]
            # update mu
            for i in range(self.N):
                new_mu[k] += self.gamma[i,k] * self.X[i]
            new_mu[k] /= post_p_sum
            # update sigma
            for i in range(self.N):
                new_sigma[k] += self.gamma[i,k] * np.reshape((self.X[i]-self.gaussian_means[k]),(self.D,1)) * np.reshape((self.X[i]-self.gaussian_means[k]), (1, self.D))
            new_sigma[k] /= post_p_sum
            # update alpha
            new_weights[k] = post_p_sum/self.N
        self.gaussian_means = new_mu
        self.gaussian_vars = new_sigma
        self.gaussian_weights = new_weights

    def __logLike(self, X, weights, mu, sigma):
        p = np.zeros((self.N, self.K))
        for k in range(self.K):
            for i in range(self.N):
                p[i,k] = multivariate_normal.pdf(X[i], mu[k,:], sigma[k,:,:])
        return np.sum(np.log(p.dot(weights)))


if __name__ == '__main__':
    X = genfromtxt("../data/TrainingData_GMM.csv", delimiter=',')
    # print(f"data shape: {X.shape}")
    # plt.scatter(X[:,0],X[:,1])
    plt.show()
    # N, D = X.shape[0], X.shape[1]
    # model = GMM(n_feature=D, n_class=4)
    # model.fit(X, train_iters=100)
    # print(model.gaussian_means)
    # print(model.gaussian_vars)
    # print(model.gaussian_weights)
    # X = genfromtxt("../data/gmm.csv", delimiter=' ')
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()
    N, D = X.shape[0], X.shape[1]
    # print(N,D)
    model = GMM(n_feature=D, n_class=4)
    model.fit(X, train_iters=60)
    pred = model.clustering(X)
    print(pred)
    markers = ['o', '+', 'x', '*']
    colors = ['r', 'b', 'y', 'g']
    for d, l in zip(X, pred):
        plt.scatter(d[0], d[1], color=colors[l], marker=markers[l])
    plt.show()


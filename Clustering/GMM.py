import numpy as np
from scipy.stats import multivariate_normal
from numpy import genfromtxt
from matplotlib import pyplot as plt

class GMM:
    def __init__(self, n_feature, n_class):
        """
        :param n_feature: D
        :param n_class: K
        """
        self.K = n_class
        self.D = n_feature

        self.gaussian_means = None  # K*D
        self.gaussian_vars = None  # K*D*D
        self.gaussian_weights = None  # K
        self.gamma = None  # shape: N*K

        self.X = None  # shape: N*D
        self.N = None

    def __init_gaussian(self):
        self.gaussian_means = np.array([[1,0],[0,1],[0,0],[1,1]]).astype(float)
        single_sigma = np.array([[1,0],[0,1]]).astype(float)
        self.gaussian_vars = np.zeros((self.K, self.D, self.D)).astype(float)
        for i in range(self.K):
            self.gaussian_vars[i,:,:] = single_sigma
        self.gaussian_weights = np.ones(self.K).astype(float)/self.K
        print(self.gaussian_means)
        print(self.gaussian_vars)
        print(self.gaussian_weights)
        self.gamma = np.zeros((self.N, self.K)).astype(float)

    def fit(self,X,train_iters):
        """
        :param X: shape: N*D
        :param train_iters: iters to train
        :return:
        """
        self.X = X
        self.N = X.shape[0]
        self.__init_gaussian()
        for iter in range(train_iters):
            self.E_step()
            self.M_step()
            loss = self.__logLike(self.X, self.gaussian_weights, self.gaussian_means, self.gaussian_vars)
            print(f"iter:{iter}, loss:{loss}")

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
    X = genfromtxt("../TrainingData_GMM.csv", delimiter=',')
    print(f"data shape: {X.shape}")
    plt.scatter(X[:,0],X[:,1])
    # plt.show()
    N, D = X.shape[0], X.shape[1]
    model = GMM(n_feature=D, n_class=4)
    model.fit(X, train_iters=200)
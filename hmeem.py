from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

class BaseHMEEM(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, em_max_iter = 1000, em_tol = 1e-4, verbose = 0):
        self.parent = None
        self.children = None
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.verbose = verbose

    def fit(self, X, Y, callbacks = []):
        ll_prev = np.inf
        for step in range(self.em_max_iter):
            self.e_step(X, Y)
            self.m_step(X, Y)
            ll = np.log(self.predict_proba(X, Y)).sum()

            if self.verbose > 0:
                print("EM: step = {}, log-likelihood = {}".format(step, ll))
            for callback in callbacks:
                callback(self, X, Y)

            if np.abs(ll - ll_prev) < self.em_tol:
                break
            else:
                ll_prev = ll

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X, Y):
        pass

    @abstractmethod
    def e_step(self, X, Y):
        if self.verbose > 1:
            print("E step: {}".format(self.id()))
        pass

    @abstractmethod
    def m_step(self, X, Y):
        if self.verbose > 1:
            print("M step: {}".format(self.id()))
        pass

    def z(self, X):
        if self.parent is not None:
            z = self.parent.g(X)[self.nth_child()] * self.parent.z(X)
        else:
            z = np.ones(X.shape[0])
        return z  # (sample size, )

    def id(self):
        if self.parent is not None:
            id = "{} -> {}".format(self.parent.id(), self.nth_child())
        else:
            id = "0"
        return id

    def nth_child(self):
        n = [i for i, child in enumerate(self.parent.children) if child is self][0]
        return n

class HMEEM(BaseHMEEM):
    def __init__(self, v, children, em_max_iter = 1000, em_tol = 1e-4, irls_max_iter = 1000, irls_tol = 1e-4, verbose = 0):
        super().__init__(em_max_iter, em_tol, verbose)
        self.v = v  # (the number of children, the number of input variables + 1)
        self.children = children
        for child in children:
            child.parent = self
        self.irls_max_iter = irls_max_iter
        self.irls_tol = irls_tol
        self.__h_i = None  # (sample size, )
        self.__h_j_i = None  # (the number of children, sample size)

    def predict(self, X):
        mu = self.g(X)[:, :, None] * np.array([child.predict(X) for child in self.children])
        mu = mu.sum(axis = 0)
        return mu  # (sample size, the number of output variables)

    def predict_proba(self, X, Y):
        p = self.g(X) * np.array([child.predict_proba(X, Y) for child in self.children])
        p = p.sum(axis = 0)
        return p  # (sample size, )

    def e_step(self, X, Y):
        super().e_step(X, Y)
        self.__h_i, self.__h_j_i = self.h(X, Y)
        for child in self.children:
            child.e_step(X, Y)

    def m_step(self, X, Y):
        super().m_step(X, Y)
        self.IRLS(X, Y)
        for child in self.children:
            child.m_step(X, Y)

    def g(self, X):
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        e = np.exp(np.dot(self.v, X_bias.T))
        g = e / e.sum(axis = 0, keepdims = True)
        return g  # (the number of children, sample size)

    def h(self, X, Y):
        if self.parent is not None:
            h_i_parent, h_j_i_parent = self.parent.h(X, Y)
            h_i = h_i_parent * h_j_i_parent[self.nth_child()]
        else:
            h_i = np.ones(X.shape[0])
        h_j_i = self.g(X) * np.array([child.predict_proba(X, Y) for child in self.children])
        h_j_i = h_j_i / h_j_i.sum(axis = 0, keepdims = True)
        return h_i, h_j_i  # (sample size, ), (the number of children, sample size)

    def IRLS(self, X, Y):
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        H = np.diag(self.__h_i)
        for step in range(self.irls_max_iter):
            g = self.g(X)
            r = g * (1 - g)
            norm = 0
            for nth_child in range(len(self.children)):
                R = np.diag(r[nth_child])
                hessian = -1 * X_bias.T.dot(H).dot(R).dot(X_bias)
                gradient = -1 * X_bias.T.dot(H).dot(g[nth_child] - self.__h_j_i[nth_child])
                # self.v[nth_child] = self.v[nth_child] + np.linalg.inv(hessian).dot(gradient)
                self.v[nth_child] = self.v[nth_child] + 0.001 * gradient
                norm += np.linalg.norm(gradient)
            if self.verbose > 2:
                print("IRLS: step = {}, |gradient| = {}".format(step, norm))
            if norm < self.irls_tol:
                break

class ExpertEM(BaseHMEEM):
    def __init__(self, U, Sigma, em_max_iter = 1000, em_tol = 1e-4, verbose = 0):
        super().__init__(em_max_iter, em_tol, verbose)
        self.U = U  # (the number of input variables, the number of output variables)
        self.Sigma = Sigma  # (the number of output variables, the number of output variables)
        self.__h = None  # (sample size, )

    def predict(self, X):
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        mu = X_bias.dot(self.U)
        return mu  # (sample size, the number of output variables)

    def predict_proba(self, X, Y):
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        D = Y - X_bias.dot(self.U)
        p = np.exp(-0.5 * (D.dot(np.linalg.inv(self.Sigma)) * D).sum(axis = 1)) / np.sqrt((2 * np.pi) ** X_bias.shape[1] * np.linalg.det(self.Sigma))
        return p  # (sample size, )

    def e_step(self, X, Y):
        super().e_step(X, Y)
        if self.parent is not None:
            h_i_parent, h_j_i_parent = self.parent.h(X, Y)
            self.__h = h_i_parent * h_j_i_parent[self.nth_child()]
        else:
            self.__h = np.ones(X.shape[0])

    def m_step(self, X, Y):
        super().m_step(X, Y)
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        H = np.diag(self.__h)
        self.U = np.linalg.inv(X_bias.T.dot(H).dot(X_bias)).dot(X_bias.T).dot(H).dot(Y)
        D = Y - X_bias.dot(self.U)
        self.Sigma = D.T.dot(H).dot(D) / H.trace()

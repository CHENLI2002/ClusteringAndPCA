import numpy as np


def b_pca(data, d):
    cov = np.cov(data)
    u, sigma, v = np.linalg.svd(cov)
    return u[:, :d], sigma[:d], v[:d, :]


def d_pca(data, d):
    data = data - np.mean(data, axis=0)
    return b_pca(data, d)


def n_pca(data, d):
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return b_pca(data, d)


def dro(data, d):
    pass


if __name__ == "__main__":
    data_2D = np.loadtxt('data/data2D.csv', delimiter=',')
    data_1000D = np.loadtxt('data/data1000D.csv', delimiter=',')
    d_2 = 1

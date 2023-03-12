import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import sparse
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :4] # we take full 4 features
Y = iris.target
C = 3 
# Normalize data
X_norm = (X - X.min())/(X.max() - X.min())
# Phân chia train : validation 
X_traning = X_norm[range(0,40)]
X_traning = np.concatenate((X_traning, X_norm[50:90]), axis=0)
X_traning = np.concatenate((X_traning, X_norm[100:140]), axis=0)
X_test = X_norm[range(40,50)]
X_test = np.concatenate((X_test, X_norm[90:100]), axis=0)
X_test = np.concatenate((X_test, X_norm[140:150]), axis=0)
Y_traning = Y[range(0,40)]
Y_traning = np.concatenate((Y_traning, Y[50:90]), axis=0)
Y_traning = np.concatenate((Y_traning, Y[100:140]), axis=0)
Y_test = Y[range(40,50)]
Y_test = np.concatenate((Y_test, Y[90:100]), axis=0)
Y_test = np.concatenate((Y_test, Y[140:150]), axis=0)
X_traning = X_traning.T
X_traning = np.concatenate((np.ones((1, 120)), X_traning), axis = 0)
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
plt.axis("off")
def convert_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A
def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A
def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20
    while count < max_count:
# mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
# stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
        return W
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))
def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)
eta = .05
d = X_traning.shape[0]
W_init = np.random.randn(X_traning.shape[0], C)
W = softmax_regression(X_traning, Y_traning, W_init, eta)
print(W[-1])

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
print("x")
print(X)
print("end")
pca = sklearnPCA(n_components=2) #2-dimensional PCA
print(Y)
transformed = pd.DataFrame(pca.fit_transform(X_norm))
plt.axis("off")
def convert_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y
print(convert_labels(Y))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import dict_learning


def spherical_kmeans(X,num_centroids, num_iter):

    #### normalize

    X=normalize(X,axis=0)
    ##### ZCA whittening
    U,S, _= np.linalg.svd(np.cov(X))
    X=U.dot(np.diag(1 / np.sqrt(S + 0.05))).dot(U.T).dot(X);

    D = np.random.normal(size=(60,num_centroids));
    D = D / np.sqrt(np.sum(D ** 2, axis=0));

    #### Spherical k_means
    for i in range(num_iter):
        S = D.T.dot(X);
        S = S * (S >= np.max(S, axis=0));
        D = X.dot(S.T) + D;
        D = D / np.sqrt(np.sum(D ** 2, axis=0));


    d,c,e=dict_learning(X=X,n_components=num_centroids,alpha=1,max_iter=num_iter)
    print(X)
    print(np.dot(d,c))
    return X,D,S


def grassman_dist(A,B):
    X = np.dot(A, A.T) - np.dot(B, B.T)
    return np.sqrt(np.sum(np.square(X)))



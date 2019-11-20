## github sources
## https://github.com/pin3da/spectral-clustering

## will input another Laplacian function, from subfunciton of megaman(socks!!)
## https://github.com/mmp2/megaman/blob/master/megaman/geometry/laplacian.py
## (home) https://github.com/mmp2/megaman

import numpy
import scipy
from sklearn.cluster import KMeans

import logging


def unnor_Lap(A):
    D = numpy.zeros(A.shape)
    w = numpy.sum(A, axis=0)
    D.flat[::len(w) + 1] = w
    print(D)
    L = D-A
    return L

## subfunction from megaman, to get normalized random walk laplacian
## https://github.com/mmp2/megaman/blob/master/megaman/geometry/laplacian.py

def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = numpy.zeros(A.shape)
    w = numpy.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)


def k_means(X, n_clusters):
    #kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    kmeans = KMeans(n_clusters=n_clusters, random_state=None)
    return kmeans.fit(X).labels_


def spectral_clustering(affinity, n_clusters, cluster_method=k_means):
    L = unnor_Lap(affinity)
    #L = laplacian(affinity)
    print(L)
    ## need to find out small eigenvalues, not largest
    ## that's why Lsym is strange, do not substract identity
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
    print(eig_vect)
    X = eig_vect.real
    print(L.dot(X[:,0]))
    rows_norm = numpy.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = cluster_method(Y, n_clusters)
    return labels

'''Implementation and of K Means Clustering
Requires : python 2.7.x, Numpy 1.7.1+'''

## url https://gist.github.com/bistaumanga/6023692

import numpy as np
from numpy import linalg as LA

def kMeans(X, K, maxIters = 10, plot_progress = None):

    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        old_centroids = centroids.copy()
        
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None: plot_progress(X, C, np.array(centroids))
            
        ## modified here, check if converge
        
        centroids = np.array(centroids)
        old_centroids = np.array(old_centroids)
        
        delta = (centroids - old_centroids)
        print(delta)
        
        if LA.norm(delta, ord=None) < 1e-16:
            print(i)
            break
        
    return np.array(centroids) , C



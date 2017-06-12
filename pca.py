from PIL import Image
import numpy as np


def pca(X):
    """Principal Component Analysis

    input: X, matrix with training data stored as flattened
           arrays in rows

    return: projection matrix (with important dimensions first),
            variance & mean."""

    # TODO: Add functionality to return only eigenvectors with k
    #       largest eigenvalues.

    # Get dimensions
    num_data, dim = X.shape

    # Center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used

        # Covariance matrix
        M = np.dot(X, X.T)
        # Eigenvalues & eigenvectors
        e, EV = np.linalg.eigh(M)
        # Compact trick
        tmp = np.dot(X.T, EV).T
        # Reverse matrix since last eigenvectors are the ones we
        # are interested in.
        V = tmp[::-1]
        # Reverse to sort eigenvalues in decreasing order
        S = np.sqrt(e)[::-1]

        for i in range(V.shape[1]):
            V[:, i] /= S

    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    # Return the projection matrix, the variance, and the mean
    return V, S, mean_X

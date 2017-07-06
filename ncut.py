from __future__ import division
import numpy as np

def ncut_graph_matrix(im, sigma_d=1e2, sigma_g=1e-2):
    """Create weights for normalized cut. The parameters are
       the weights for pixel distance and pixel similarity.
    """
    m, n = im.shape[:2]
    N = n * m

    # Normalize and create feature victro of rgb or grayscale
    if len(im.shape) == 3:
        for i in range(3):
            im[:, :, i] = im[:, :, i].max()
        vim = im.reshape((-1, 3))
    else:
        im = im / im.max()
        vim = im.flatten()

    # x, y coordinates for distance calculation
    xx, yy = np.meshgrid(range(m), range(n))
    x, y = xx.flatten(), yy.flatten()

    # Create matrix with edge weights
    W = np.zeros((N, N), 'f')
    for i in range(N):
        for j in range(i, N):
            d = (x[i] - x[j] ** 2 + y[i] - y[j] ** 2)
            W[i, j] = W[j, i] = np.exp(-1.0 * np.sum((vim[i] - vim[j]) ** 2)
                                       / sigma_g)
    return W

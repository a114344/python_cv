from homography import Haffine_from_points
import numpy as np
from scipy import ndimage


def image_in_image(im1, im2, tp):
    """Put im1 in im2 with an affine transformation such that the corners

       are as close to tp as possible. tp are homogenous and counterclockwise

       from top left.
    """

    # Points to warp from
    m, n = im1.shape[:2]
    fp = np.array([[0, m, m, 0],
                   [0, 0, n, n],
                   [1, 1, 1, 1]])

    # compute affine transformation and apply
    H = Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2],
                                     (H[0, 2], H[1, 2]),
                                     im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha) * im2 + alpha * im1_t


def alpha_for_triangle(points, m, n):
    """Creates alpha map of size (m, n) for a triangle defined by points

       given in normalized homogenous format.

    """
    alpha = np.zeros(m, n)
    for i in range(np.min(points[0]), np.max(points[0])):
        for j in range(np.min(points[1]), np.max(points[1])):
            x = np.linalg.solve(points[i, j, 1])
            if np.min(x) > 0:
                alpha[i, j] = 1

    return alpha

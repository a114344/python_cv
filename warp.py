from homography import Haffine_from_points
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.delaunay as md


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


def triangulate_points(x, y):
    """Delauney triangulation of 2d points.
    """
    centers, edges, tri, neighbors = md.delaunay(x, y)

    return tri


def pw_affine(fromim, toim, fp, tp, tri):
    """Warp particular patches from an image.

       Inputs:
              fromim = image to Warp
              toim = destination image
              fp = from points in hom. coordinates
              tp = to pints in hom coordinates
              tri = triangualation
    """

    im = toim.copy()

    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3

    # Create an image and warp to (needed if iterate colors)
    im_t = np.zeros(im.shape, 'uint8')

    for t in tri:
        # Compute affine transoformation
        H = Haffine_from_points(tp[:, t], fp[:, t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(fromim[:, :, col],
                                                           H[:2, :2],
                                                           (H[0, 2], H[1, 2]),
                                                           im.shape[:2])
        else:
            im_t = ndimage.affine_transform(fromim,
                                            H[:2, :2],
                                            (H[0, 2], H[1, 2]),
                                            im.shape[:2])
        # Alpha for triangle
        alpha = alpha_for_triangle(tp[:, t], im.shape[0], im.shape[1])

        # Add triangle to image
        im[alpha > 0] = im_t[alpha > 0]

    return im


def plot_mesh(x, y, tri):
    """Plot triangles.
    """
    for t in tri:
        # Add first point to end
        t_ext = [t[0], t[1], t[2], t[0]]
        plt.plot(x[t_ext], y[t_ext], 'r')

    return True

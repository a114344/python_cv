import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import filters


def compute_harris_response(img, sigma=3):
    """Compute Harris corner detector response function for each pixel in a
        grayscale image.
    """
    # derivatives
    imx = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)

    # Compute components of Harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_poins(harrisimg, min_dist=10, threshold=0.1):
    """Return corners for a Harris response image. min_dist
       is the minumum number of pixels separating corners and image
       boundary.
    """

    # find top corner candidates above threshold
    corner_threshold = harrisimg.max() * threshold
    harrisimg_t = (harrisimg > corner_threshold) * 1

    # get coordinates of candidates
    coords = array(harrisimg_t.nonzero()).T

    # ...and their values
    candidate_values = 

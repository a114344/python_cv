"""
Collection of basic image processing functions.
"""
from __future__ import print_function
import numpy as np
import os
from PIL import Image


def get_imlist(path):
    """Returns a list of filenames for all
    jpg images in target directory."""

    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('jpg')]


def imresize(img, size):
    """Resize an image using PIL."""

    pil_im = Image.fromarray(np.uint8(img))

    return np.array(pil_im.resize(size))


def histeq(im, bins=256):
    """Histogram equalization of a greyscale image."""

    # get image Histogram
    imhist, bins = np.histogram(im.flatten(), bins, normed=True)
    # Cumulative density function
    cdf = imhist.cumsum()
    # Normalize
    cdf = 255 * cdf / cdf[-1]
    # Use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    """Compute and output the average of a list of images."""

    # open first image and make into an array of type float
    averageim = np.array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)

    return np.array(averageim, np.uint8)

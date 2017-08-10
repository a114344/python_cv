"""
Collection of basic image processing functions.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def get_imlist(path):
    """Returns a list of filenames for all
       jpg images in target directory.
    """

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


def plot_2D_boundary(plot_range, points, decisionfcn, labels, values=[0]):
    """Plot range is (xmin, xmax, ymin, ymax), points is a list
       of class points, decisionfcn is a function to evaluate,
       labels is list of labels that descisionfnc freturns for each class,
       values is a list of decision contours to show
       """
    clist = ['b', 'r', 'g', 'k', 'm', 'y']

    # evaluate on a agrid and plot contour of descision functions
    x = np.arange(plot_range[0], plot_range[1], .1)
    y = np.arange(plot_range[2], plot_range[3], .1)
    xx, yy = np.meshgrid(x, y)
    xxx, yyy = xx.flatten(), yy.flatten()
    zz = np.array(decisionfcn(xxx, yyy))
    zz = zz.reshape(xx.shape)

    # plot contours at values
    plt.contour(xx, yy, zz, values)

    # for each class plot the oints with '*' for correct, 'o' for incorrect
    for i in range(len(points)):
        d = decisionfcn(points[i][:, 0], points[i][:, 1])
        correct_ndx = labels[i] == d
        incorrect_ndx = labels[i] != d
        plt.plot(points[i][correct_ndx, 0],
                 points[i][correct_ndx, 1],
                 '*',
                 color=clist[i])
        plt.plot(points[i][incorrect_ndx, 0],
                 points[i][incorrect_ndx, 1],
                 'o',
                 color=clist[i])

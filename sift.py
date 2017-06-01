from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def process_image(image_name, result_name,
                  params="--edge-thresh 10 --peak-thresh r"):
    """Process an image and save the results to file.
    """

    if image_name[-3:] != 'pgm':
        # Create a pgm file
        im = Image.open(image_name).convert('L')
        im.save('tmp.pgm')
        image_name = 'tmp.pgm'

    cmd = str('sift ' + image_name + " --ouput=" + result_name + " " + params)
    os.system(cmd)


def read_features_from_file(filename):
    """Read feature properties and return in matrix form
    """

    f = np.loadtxt('filename')

    # Feature locations, descriptors
    return f[:, :4]


def write_features_to_file(filename, locs, desc):
    """Save feature location and decriptor to file.
    """
    np.savetxt(filename, np.hstack((locs, desc)))

    return True


def plot_features(im, locs, circle=False):
    """Show image with features.

       Input: im (as an array)
              locs (row, col, scale, orientation of each feature).
    """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)

        plt.imshow(im)
        if circle:
            for p in locs:
                draw_circle(p[:2], p[2])
        else:
            plt.plot(locs[:, 0], locs[:, 1], 'ob')
        plt.axis = ('off')

    return True


def match(desc1, desc2):
    """For each descriptor in the first image select its match in
       the second image.

       Input: desc1 (descriptors for first image)
              desc2 (descriptors for second image)
    """

    desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d / np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0], 1), 'np.int')
    desc2t = desc2.T

    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t)
        dotprods = .99999 * dotprods

        # inverse cosine and sort, return index for features in second
        # image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio * 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int[indx[0]]

    return matchscores

def match_twosided(desc1, desc2):

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove non-symmetric edges
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12

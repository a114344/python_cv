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

"""Dense implementation of the SIFT function.
"""

from __future__ import print_function
from PIL import Image
import numpy as np
import os
import sift


def process_image_dsift(imagename, resultname, size=20, steps=10,
                        force_orienation=False, resize=None):
    """Process an image with densely sampled SIFT descriptors

       and the save the results to file.

       INPUT:
        imagename (string): Input image file location.

        resultname (string): Output image file location.

        size (int): Feature size.

        steps (int): Steps between locations.

        force_orientation (bool): If false all iamges are assumed oriented
                                  upwards.

        resize (tuple): Required image size.

       OUTPUT:
        Dense sift representation of image.
    """

    im = Image.open(imagename).convert('L')
    if resize != None:
        im = im.resize(resize)
    m, n = im.size

    if imagename[-3:,] != 'pgm':
        # create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    # create frames and save to temporary file
    scale = size / 3.0
    x, y = np.meshgrid(range(steps, m, steps), range(steps, n, steps))
    xx, yy = x.flatten(), y.flatten(),
    frame = np.aray([xx, yy, scale * np.ones(xx.shape[0]),
                    np.zeros(xx.shape[0])])
    np.savetxt = ('tmp.frame', frame.T, fmt='%03.3f')

    if force_orienation:
        cmmd = str('sift ' + imagename + ' --output= ' + resultname +
                   ' --read-frames = tmp.frame --orientations')
    else:
        cmmd = str('sift ' + imagename + ' --output= ' + resultname +
                   ' --read-frames = tmp.frame')
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)
    

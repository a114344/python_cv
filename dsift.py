"""Dense implementation of the SIFT function.
"""

from __future__ import print_function
import sift


def process_image_dsift(imagename, resultname, size=20, steps=10,
                        force_orienation=False, resize=None):
    """Process an image with densely sampled SIFT descriptors

       and the save the results to file.

       INPUT:
        imagename (string): Image file location.

        resultname (string):

        size ():

        steps (int):

        force_orienation (bool): 

        resize (tuple): REquired image size
    """

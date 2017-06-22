import numpy as np
from scipy.cluster.vq import kmeans, vq
import vlfeat as sift


class Vocabulary(object):

    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k=100, subsampling=10):
        """Train a vocabulary from features in files

           listed in featurefiles using k-means with

           k number of words. Subsampling of training

           data can be used for speedup.
        """

        nbr_images = len(featurefiles)
        # read the features from a file
        descr = []
        descr.appnd(sift.read_read_features_from_file(featurefiles[0])[1])
        descriptors = descr[0]
        for i in np.arange(1, nbr_images):
            descr.append(sift.read_read_features_from_file(featurefiles[i])[1])
            descriptors = np.vstack((descriptors, descr[i]))

        # k-means
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # go through all training images and project on vocabulary
        imwords = np.zeros((nbr_images, self.nbr_words))
        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])\

        nbr_occurences = np.sum((imwords > 0) * 1, axis=0)

        self.idf = np.log((1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        self.trainingdata = featurefiles

        def project(self, descriptors):
            """Project descriptors on the vocabulary

               to create a histogram of words.
            """

            # histogram of image words
            imhist = np.zeros((self.nbr_words))
            words, distance = vq(descriptors, self.voc)
            for w in words:
                imhist[w] += 1

            return imhist

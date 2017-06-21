from scipy.cluster.vq import *
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

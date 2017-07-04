from __future__ import print_function
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

import bayes

def build_bayes_graph(im, labels, sigma=1e2, kappa=2):
    """Build a graph from 4-neighborhood of pixels.

       Foreground and background is determined from

       labels (1 for forground, -1 for background, 0 otherwise)

       and is modelled with naive bayes classifers.
    """
    m, n = im.shape[:2]

    # RGB vector version (one pixel per row)
    vim = im.reshape((-1, 3))

    # RGB for foreground and background
    foreground = im[labels == 1].reshape((-1, 3))
    background = im[labels == -1].reshape((-1, 3))
    train_data = [foreground, background]

    # Train naive bayes classifier
    bc = bayes.BayesClassifier()
    bc.train(train_data)

    # get probabilities
    bc_labels, prob = bc.classify(vim)
    

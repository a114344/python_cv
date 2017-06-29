import nump as np


class KnnClassifier(object):

    def __init__(self, labels, samples):
        self.labels = labels
        self.samples = samples

    def classify(self, point, k=3):
        """Classify a point against k nearest in
           the training data  & return a label.
        """

        # compute distance to all training points
        dist = np.array([L2dist(point, s) for s in self.samples])

        # sort
        ndx = dist.argsort()

        # use dict to store nearest k points
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1

        return np.max(votes)

    def L2dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    

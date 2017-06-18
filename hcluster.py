from itertools import combinations
import numpy as np


class ClusterNode(object):
    def __init__(self, vec, left, right, distance=0.0, count=1):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.count = count

    def extract_clusters(self, dist):
        """Extract list of sub-tree clusters from hcluster

           tree with distance < dist.
        """
        if self.distance < dist:
            return [self]
        return (self.left.extract_clusters(dist) +
                self.right.extract_clusters(dist))

    def get_cluster_elements(self):
        """Return ids for elements in a cluster sub-tree.
        """
        return (self.left.get_cluster_elements() +
                self.right.get_cluster_elements())

    def get_height(self):
        """Return the height of a node, height is sum

           of each branch.
        """
        return (self.left.get_height() + self.right.get_height())

    def get_depth(self):
        """Return the depth of a node, depth is max of

           each child plus own distance.
        """
        return np.max(self.left.get_depth(),
                      self.right.get_depth() +
                      self.distance)


class ClusterLeafNode(object):
    def __init__(self, vec, id):
        self.vec = vec
        self.id = id

    def extract(self, dist):
        return [self]

    def get_cluster_elements(self):
        return [self.id]

    def get_height(self):
        return 1

    def get_depth(self):
        return 0


def L2dist(v1, v2):
    

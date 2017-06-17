from itertools import combinations


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
        

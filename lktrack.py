import cv2
import numpy as np


lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))
subpix_params = dict(zeroZone=(-1, 1), winsize=(10, 10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                               20, 0.03))
feature_params = dict(maxCorners=500, quality_level=0.01, minDistance=10)


class LKTracker(object):
    """Class for Lucas-Kinkade tracking with pyramidal

       optical flow.
    """

    def __init__(self, imnames):
        """Initialiaze with a set of image names.
        """
        self.imnames = imnames
        self.features = []
        self.tracks = []
        self.current_frame = 0

    def detect_points(self):
        """Detect 'good features to track' (corners) in the

           current frame using sub-pixel accuracy.
        """

        # load image and create grayscale
        self.image = cv2.imread(self.imnames[self.current_frame])
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # search for good points
        features = cv2.goodFeaturesToTrack(self.gray,  **feature_params)

        # refine corner locations
        cv2.cornerSubPix(self.gray, features, **subpix_params)

        self.features = features
        self.tracks = [[p] for p in features.reshape((-1, 2))]

        self.prev_gray = self.gray

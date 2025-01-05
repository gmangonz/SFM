import cv2
import numpy as np


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CONFIG(DotDict):

    nfeatures = 0
    nlayers = 6  # 6, 3
    contrastThreshold = 0.04  # 0.04
    edgeThreshold = 40  # 40, 5
    sigma = 0.5  # 0.5, 1.6
    threshold = 0.75
    ransacReprojThreshold = 1.0
    confidence = 0.999
    RGB = True
    clahe = True
    NMS = False
    NMS_dist = 5  # 5, 3
    clipLimit = 10.0  # 40
    tileGridSize = (8, 8)

    def __init__(self, **kwargs):
        super(CONFIG, self).__init__(**kwargs)


siftDetector = cv2.SIFT.create(
    nOctaveLayers=CONFIG.nlayers,
    nfeatures=CONFIG.nfeatures,
    contrastThreshold=CONFIG.contrastThreshold,
    edgeThreshold=CONFIG.edgeThreshold,
    sigma=CONFIG.sigma,
)


K = np.array([[3053, 0, 2016], [0, 3053, 1512], [0, 0, 1]])

W_ratio = 1080 / 4032
K[0, 0] = K[0, 0] * W_ratio
K[1, 1] = K[1, 1] * W_ratio
K[0, 2] = K[0, 2] * W_ratio

H_ratio = 1920 / 3024
K[1, 2] = K[1, 2] * H_ratio

K = np.array([[1080, 0, 540], [0, 1080, 960], [0, 0, 1]])

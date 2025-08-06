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
    method = "dijkstra"


K = np.array([[1080, 0, 540], [0, 1080, 960], [0, 0, 1]])

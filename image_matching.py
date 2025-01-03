from dataclasses import dataclass

import cv2
from torch.utils.data import Dataset

from utils import ImageData


@dataclass
class DMatch:
    queryIdx: int
    trainIdx: int
    imgIdx: int
    distance: float


class MatchingDataset(Dataset):
    def __init__(self, possible_pairs: list[tuple[int, int]], image_data: list[ImageData], threshold: float):
        super(MatchingDataset, self).__init__()

        self.possible_pairs = possible_pairs
        self.image_data = image_data
        self.threshold = threshold

    def __getitem__(self, idx: int):

        camera_0, camera_1 = self.possible_pairs[idx]
        image_0 = self.image_data[camera_0]
        image_1 = self.image_data[camera_1]

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(image_0.features, image_1.features, 2)

        matches: list[cv2.DMatch] = []
        for m, n in knn_matches:
            if m.distance < self.threshold * n.distance:
                matches.append(DMatch(m.queryIdx, m.trainIdx, m.imgIdx, m.distance))

        return camera_0, camera_1, matches

    def __len__(self):
        return len(self.possible_pairs)

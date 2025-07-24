from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from sfm.config import CONFIG
from sfm.utils import ImageData


@dataclass
class DMatch:
    queryIdx: int
    trainIdx: int
    imgIdx: int
    distance: float


class MatchingDataset(Dataset):
    def __init__(
        self,
        possible_pairs: list[tuple[int, int]],
        image_data: list[ImageData],
        threshold: float,
    ):
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

        # Geometric verification
        pts1 = np.float32([image_0.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([image_1.keypoints[m.trainIdx].pt for m in matches])

        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0)
        matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]

        return camera_0, camera_1, matches

    def __len__(self):
        return len(self.possible_pairs)


def get_image_matcher_loader(image_data: list[ImageData]) -> DataLoader:
    """
    Data loader to iterate through all image pairs.

    Parameters
    ----------
    image_data : list[ImageData]
        List of image data that will be iterated through using r-length combinations.

    Returns
    -------
    DataLoader
        Data loader.
    """

    pairs = list(combinations(range(len(image_data)), 2))
    dataset = MatchingDataset(possible_pairs=pairs, image_data=image_data, threshold=CONFIG.threshold)
    pair_loader = DataLoader(
        dataset,
        num_workers=10,
        shuffle=False,
        pin_memory=False,
        collate_fn=lambda x: x,
        prefetch_factor=2,
    )
    return pair_loader

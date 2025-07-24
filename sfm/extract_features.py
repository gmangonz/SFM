from pathlib import Path

import cv2
import natsort
import numpy as np
from torch.utils.data import DataLoader, Dataset

from sfm.config import CONFIG
from sfm.utils import SIFT, KeyPoint


class ImageDataset(Dataset):

    def __init__(self, image_dir: str, grayscale: bool):
        image_paths = [str(path) for i, path in enumerate(Path(image_dir).glob("*.png"))]

        self.image_paths = natsort.natsorted(image_paths)
        self.grayscale = grayscale

    def __getitem__(self, idx: int):

        path = self.image_paths[idx]

        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read image {path}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if self.grayscale else image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch) -> tuple[np.ndarray, np.ndarray, list[KeyPoint], np.ndarray]:
    x: np.ndarray = batch[0]

    clahe, kpts, descriptors = SIFT(x)
    kpts = kpts.tolist()
    kpts = [KeyPoint(kpt.pt, kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id) for kpt in kpts]

    return x.astype(np.float32) / 255.0, clahe, kpts, descriptors


def get_image_loader(image_dir: str) -> DataLoader:
    """
    Get image loader that iterates though image directory to extract
    clahe image, keypoints, and descriptors

    Parameters
    ----------
    image_dir : str
        Image directory

    Returns
    -------
    DataLoader
        Data loader.
    """

    dataset = ImageDataset(image_dir, grayscale=not CONFIG.RGB)
    loader = DataLoader(
        dataset,
        num_workers=10,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return loader

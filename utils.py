import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import gtsam.utils.plot
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import torch

__cwd__ = Path(os.path.dirname(__file__))


from config import CONFIG, K, siftDetector

formatter = logging.Formatter(fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


class SingleCamera:
    def __init__(
        self,
        K: np.ndarray,
        distortion: np.ndarray,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
    ):
        self.K = K
        self.dist = distortion

        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.P = self.K @ np.concatenate((self.R, self.t.reshape(3, 1)), axis=1)

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        uv_undistort = cv2.undistortImagePoints(
            uv.reshape(-1, 2),
            cameraMatrix=self.K,
            distCoeffs=self.dist,
        )
        return np.reshape(uv_undistort, uv.shape)

    def get_gtsam_pose(self):
        return gtsam.Pose3(gtsam.Rot3(self.R), np.reshape(self.t, (3, 1)))


class StereoCamera:
    def __init__(self, R_1: np.ndarray, t_1: np.ndarray, R_0: np.ndarray = None, t_0: np.ndarray = None):
        self.update(
            R_1=R_1,
            t_1=t_1,
            R_0=R_0,
            t_0=t_0,
        )
        self.pts_3D = None

    def update(self, R_1: np.ndarray, t_1: np.ndarray, R_0: np.ndarray = None, t_0: np.ndarray = None):
        self.cam_0 = SingleCamera(
            K=K,
            distortion=np.zeros(5),
            R=R_0,
            t=t_0,
        )

        self.cam_1 = SingleCamera(
            K=K,
            distortion=np.zeros(5),
            R=R_1,
            t=t_1,
        )

    def triangulate(self, uv_left: np.ndarray, uv_right: np.ndarray) -> np.ndarray:

        old_shape = [*uv_left.shape[:-1], 3]
        uv_left, uv_right = uv_left.reshape(-1, 2), uv_right.reshape(-1, 2)

        uv_left = self.cam_0.undistort_points(uv_left)
        uv_right = self.cam_1.undistort_points(uv_right)

        xyz_h = cv2.triangulatePoints(
            self.cam_0.P,
            self.cam_1.P,
            uv_left.T,
            uv_right.T,
        )
        xyz_h = xyz_h.T
        xyz = xyz_h[..., :-1] / xyz_h[..., -1:]
        self.pts_3D = np.reshape(xyz, old_shape)
        return self.pts_3D

    def get_gtsam_poses(self):
        return self.cam_0.get_gtsam_pose(), self.cam_1.get_gtsam_pose()

    def __getitem__(self, key) -> SingleCamera:
        return getattr(self, "cam_0" if key == 0 else ("cam_1" if key == 1 else str(key)))


@dataclass
class KeyPoint:
    pt: tuple[float, float]
    size: float
    angle: float
    response: float
    octave: int
    class_id: int


@dataclass
class ImageData:
    image: np.ndarray
    rgb: np.ndarray
    filename: str
    keypoints: list[KeyPoint]
    features: np.ndarray
    clahe_img: np.ndarray

    def __post_init__(self):
        keypoints = [kpt.pt for kpt in self.keypoints]
        self.keypoints_np = np.asarray(keypoints)


def NMS(keypoints: list[cv2.KeyPoint], radius: int) -> np.ndarray:

    points = np.array([kp.pt for kp in keypoints])  # shape: (N, 2)
    responses = np.array([kp.response for kp in keypoints])  # shape: (N,)

    sorted_indices = np.argsort(-responses)
    sorted_points = points[sorted_indices]

    keep_mask = np.ones(len(keypoints), dtype=bool)

    for i in range(len(sorted_points)):
        if not keep_mask[i]:
            continue

        distances = np.linalg.norm(sorted_points[i] - sorted_points[i + 1 :], axis=1)
        suppression_mask = distances < radius
        keep_mask[i + 1 :][suppression_mask] = False

    return sorted_indices[keep_mask]


def SIFT(img: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray[cv2.KeyPoint], np.ndarray]:

    if isinstance(img, torch.Tensor):
        img = img.numpy().astype(np.uint8)

    if not CONFIG.RGB and CONFIG.clahe:
        clahe = cv2.createCLAHE(clipLimit=CONFIG.clipLimit, tileGridSize=CONFIG.tileGridSize)
        clahe_img = clahe.apply(img)
    else:
        clahe_img = img.copy()

    kpts, descriptors = siftDetector.detectAndCompute(image=clahe_img, mask=None)

    if CONFIG.NMS:
        mask = NMS(kpts, radius=CONFIG.NMS_dist)
        kpts = np.array(kpts)[mask]
        descriptors = np.array(descriptors)[mask]

    return clahe_img, np.asarray(kpts), np.asarray(descriptors)


def get_src_dst_pts(
    camera_0: int,
    camera_1: int,
    queryIdx: np.ndarray,
    trainIdx: np.ndarray,
    image_data: list[ImageData],
) -> tuple[np.ndarray, np.ndarray]:

    image_0 = image_data[camera_0]
    image_1 = image_data[camera_1]

    src_pts = image_0.keypoints_np[queryIdx]
    dst_pts = image_1.keypoints_np[trainIdx]
    assert len(src_pts) == len(dst_pts), "Length of queryIdx and trainIdx need to match!"
    return src_pts, dst_pts


def to_cv2KeyPoint(data: KeyPoint):
    return cv2.KeyPoint(data.pt[0], data.pt[1], data.size, data.angle, data.response, data.octave, data.class_id)


def estimate_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    # Expects Homogenous points

    num_points = min(points1.shape[0], points2.shape[0])
    A = np.empty((num_points, 9))

    x1_ = points1.repeat(3, axis=1)
    x2_ = np.tile(points2, (1, 3))

    A = x1_ * x2_

    u, s, v = np.linalg.svd(A)
    F = v[-1, :].reshape((3, 3), order="F")

    u, s, v = np.linalg.svd(F)
    F = u.dot(np.diag(s).dot(v))
    F = F / F[-1, -1]
    return F


def sampson_error(F: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    # Expects Homogenous points

    F_src: np.ndarray = np.dot(F, points1.T)
    Ft_dst: np.ndarray = np.dot(F.T, points2.T)

    dst_F_src = np.sum(points2 * F_src.T, axis=1)

    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


def fundamental_ransac(
    points1: np.ndarray, points2: np.ndarray, iterations=5000, threshold=0.1
) -> tuple[np.ndarray, np.ndarray]:

    points1 = points1.squeeze()
    points2 = points2.squeeze()

    if points1.shape[1] == 2:
        points1 = cv2.convertPointsToHomogeneous(points1)[:, 0, :]
        points2 = cv2.convertPointsToHomogeneous(points2)[:, 0, :]

    best_inliers = None
    most_inliers = 0

    for _ in range(iterations):

        rand_indexes = np.random.choice(points1.shape[0], 8, replace=False)
        rand_points1 = points1[rand_indexes]
        rand_points2 = points2[rand_indexes]

        # Count inliers (use fact that a*F*b.T = 0)
        F = estimate_fundamental_matrix(rand_points1, rand_points2)

        err = sampson_error(F, points1, points2)
        mask = err < threshold
        num_inliers = np.sum(mask)

        if num_inliers > most_inliers:
            most_inliers = num_inliers
            best_inliers = (points1[err < threshold], points2[err < threshold])

    F = estimate_fundamental_matrix(*best_inliers)
    err = sampson_error(F, points1, points2)
    return F, err < threshold


def get_optimal_E(
    E: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, K: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    _, R_out, t_out, mask = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    inliers = sum(mask)
    return R_out, t_out, mask, inliers


def find_optimal_E(
    E: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, K: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    most_inliers = 0
    R_out, t_out, mask_out = None, None, None
    for e in range(0, E.shape[0], 3):

        R_, t_, mask_, inliers = get_optimal_E(
            E=E[e : e + 3, :].copy(), src_pts=src_pts.copy(), dst_pts=dst_pts.copy(), K=K, mask=mask.copy()
        )
        if inliers > most_inliers:
            most_inliers = inliers
            R_out, t_out, mask_out = R_, t_, mask_

    return R_out, t_out, mask_out


def compute_essential_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E, mask_out = cv2.findEssentialMat(
        src_pts,
        dst_pts,
        K,
        method=cv2.RANSAC,
        prob=CONFIG.confidence,
        threshold=CONFIG.ransacReprojThreshold,
    )
    inliers = mask_out.ravel().astype(bool) if mask_out is not None else np.zeros(src_pts.shape[0], dtype=bool)
    return E, mask_out, inliers


def get_pose(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the relative pose between 2 cameras given the source and
    destination points in the respective image planes.

    Given ALL the matches between the 2 images, cv2.findEssentialMat will
    filter the matches that optimize the essential matrix. Then, cv2.recoverPose
    will further filter the matches to those that are infront of the camera.

    Parameters
    ----------
    src_pts : np.ndarray
        Source points.
    dst_pts : np.ndarray
        Destination points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Essential matrix, mask, inliers, rotation, and translation.
    """
    E, mask_out, inliers = compute_essential_matrix(src_pts, dst_pts)
    # if E.shape[0] > 3 and E.shape[0] % 3 == 0:
    #     R_out, t_out, mask_out = find_optimal_E(E, src_pts, dst_pts, K, mask)
    # else:
    _, R_out, t_out, _ = cv2.recoverPose(E, src_pts, dst_pts, K)  # , mask=mask_out)
    return E, mask_out, inliers, R_out, t_out


def get_pnp_pose(
    pts_3D: np.ndarray, pts_2D: np.ndarray, dist_coeffs: np.ndarray, mask: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if mask is None:
        mask = np.ones(len(pts_3D), dtype=bool)

    if len(pts_3D[mask]) < 4:
        return None, None, None

    initial_indices = np.arange(len(pts_3D))
    success, R, t, inliers_indices = cv2.solvePnPRansac(
        pts_3D[mask],
        pts_2D[mask],
        K.astype(np.float32),
        dist_coeffs,
        reprojectionError=1.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    masked_indices = initial_indices[mask]
    result_indices = masked_indices[inliers_indices]

    if success:
        R, _ = cv2.Rodrigues(R)
        return R, t, np.isin(initial_indices, result_indices, assume_unique=True)

    return None, None, None


def get_edge_relation(reference_edge: tuple[int, int], new_edge: tuple[int, int]) -> tuple[str, int]:

    if not isinstance(reference_edge, tuple):
        raise ValueError(f"Expected reference_edge to be a tuple, got {type(reference_edge)} instead.")
    if not isinstance(new_edge, tuple):
        raise ValueError(f"Expected new_edge to be a tuple, got {type(new_edge)} instead.")

    shared_cameras = set(reference_edge).intersection(new_edge)
    assert len(shared_cameras) == 1, f"Expected one shared camera between {reference_edge} and {new_edge}."
    shared_camera = shared_cameras.pop()

    index = reference_edge.index(shared_camera)
    edge_loc = "successor" if shared_camera == new_edge[0] else "predecessor"
    return edge_loc, index


def pose_to_matrix(pose):
    return np.asanyarray(pose.matrix())


def visualize_graph(
    estimates,
    colors: dict[gtsam.Symbol, np.ndarray],
    pt_scale: int = 3,
    title: str = "SFM + Trajectory",
    debug_key: str = "",
):

    fig = plt.figure(1, figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    landmarks = []
    display_colors = []
    camera_frames = []

    for key in estimates.keys():
        symbol = gtsam.Symbol(key)

        if symbol.string()[0] == "l":  # Landmark
            landmark_pos = estimates.atPoint3(key)
            landmarks.append([*landmark_pos])
            if symbol.string() == debug_key:
                display_colors.append([255, 0, 0])
            else:
                display_colors.append(colors[key])

        elif symbol.string()[0] == "c":  # Camera
            pose = estimates.atPose3(key)
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            camera_frame.transform(pose_to_matrix(pose))
            camera_frames.append(camera_frame)

    landmarks = np.array(landmarks)
    display_colors = np.array(display_colors) / 255.0

    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        landmarks[:, 2],
        c=display_colors,
        s=pt_scale,
    )

    gtsam.utils.plot.plot_trajectory(1, estimates, scale=3, title=title)
    gtsam.utils.plot.set_axes_equal(1)

    ax = plt.gca()
    ax.view_init(elev=-75, azim=-90)

    save_pcd(landmarks, display_colors, camera_frames, title=title)
    plt.show()


def save_pcd(landmarks: np.ndarray, display_colors: np.ndarray, camera_frames: list[np.ndarray], title: str = "pcd"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(landmarks)
    point_cloud.colors = o3d.utility.Vector3dVector(display_colors)

    output_filename = "_".join(title.split())
    output_filename = f"{output_filename}.ply"
    o3d.io.write_point_cloud(os.path.join(__cwd__, output_filename), point_cloud)

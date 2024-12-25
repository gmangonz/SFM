import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gtsam
import matplotlib.pyplot as plt
import natsort
import numpy as np
import open3d as o3d

__cwd__ = Path(os.path.dirname(__file__))

K = np.array([[3053, 0, 2016], [0, 3053, 1512], [0, 0, 1]])

W_ratio = 1080 / 4032
K[0, 0] = K[0, 0] * W_ratio
K[1, 1] = K[1, 1] * W_ratio
K[0, 2] = K[0, 2] * W_ratio

H_ratio = 1920 / 3024
K[1, 2] = K[1, 2] * H_ratio

K = np.array([[1080, 0, 540], [0, 1080, 960], [0, 0, 1]])


class CONFIG:
    nfeatures: int = 0
    nlayers: int = 3  # 6, 3
    contrastThreshold: float = 0.04  # 0.04
    edgeThreshold: int = 10  # 40, 5
    sigma: float = 1.6  # 0.5, 1.6
    threshold: float = 0.75

    ransacReprojThreshold: float = 1.0
    # maxIters: int = 2000
    confidence: float = 0.999

    NMS: bool = True
    NMS_dist: int = 3  # 5, 3

    RGB: bool = False

    clahe: bool = True
    clipLimit: float = 2.0  # 40
    tileGridSize: tuple[int, int] = (8, 8)


siftDetector = cv2.SIFT.create(
    nOctaveLayers=CONFIG.nlayers,
    nfeatures=CONFIG.nfeatures,
    contrastThreshold=CONFIG.contrastThreshold,
    edgeThreshold=CONFIG.edgeThreshold,
    sigma=CONFIG.sigma,
)


def get_buddah_images() -> list[np.ndarray]:

    image_paths = [str(path) for i, path in enumerate((__cwd__ / "buddha_images").glob("buddha_*.png")) if i not in []]
    image_paths = natsort.natsorted(image_paths)

    images_COLOR = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_paths]
    images_GRAY = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY) for image_path in image_paths]

    images = [
        cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) for image in (images_COLOR if CONFIG.RGB else images_GRAY)
    ]
    return images, images_COLOR, image_paths


def NMS(keypoints: list[cv2.KeyPoint], descriptors: np.ndarray, radius=10) -> np.ndarray:

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


def SIFT(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not CONFIG.RGB and CONFIG.clahe:
        clahe = cv2.createCLAHE(clipLimit=CONFIG.clipLimit, tileGridSize=CONFIG.tileGridSize)
        clahe_img = clahe.apply(img)
    else:
        clahe_img = img

    kpts, descriptors = siftDetector.detectAndCompute(image=clahe_img, mask=None)

    if CONFIG.NMS:
        mask = NMS(kpts, descriptors, radius=CONFIG.NMS_dist)
        kpts = np.array(kpts)[mask]
        descriptors = np.array(descriptors)[mask]

    return clahe_img, kpts, descriptors


def BFMatcher(
    kpts_1: list[cv2.KeyPoint],
    descriptors_1: np.ndarray,
    kpts_2: list[cv2.KeyPoint],
    descriptors_2: np.ndarray,
    threshold: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_1, descriptors_2, 2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    src_pts_pre_match: np.ndarray = np.float32([kp.pt for kp in kpts_1]).reshape(-1, 1, 2)
    dst_pts_pre_match: np.ndarray = np.float32([kp.pt for kp in kpts_2]).reshape(-1, 1, 2)

    src_idx_match = np.array([m.queryIdx for m in good], dtype=int)
    dst_idx_match = np.array([m.trainIdx for m in good], dtype=int)

    src_pts: np.ndarray = np.float32([kpts_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts: np.ndarray = np.float32([kpts_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_pts_pre_match, dst_pts_pre_match, src_idx_match, dst_idx_match, src_pts, dst_pts


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
        self.rvec, _ = cv2.Rodrigues(self.R)
        self.t = t if t is not None else np.zeros(3)
        self.P = self.K @ np.concatenate((self.R, self.t.reshape(3, 1)), axis=1)

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        uv_undistort = cv2.undistortImagePoints(
            uv.reshape(-1, 2),
            cameraMatrix=self.K,
            distCoeffs=self.dist,
        )
        return np.reshape(uv_undistort, uv.shape)


class StereoCamera:
    def __init__(self, R_1: np.ndarray, t_1: np.ndarray, R_0: np.ndarray = None, t_0: np.ndarray = None):

        self.left_cam = SingleCamera(
            K=K,
            distortion=np.zeros(5),
            R=R_0,
            t=t_0,
        )

        self.right_cam = SingleCamera(
            K=K,
            distortion=np.zeros(5),
            R=R_1,
            t=t_1,
        )

    def triangulate(self, uv_left: np.ndarray, uv_right: np.ndarray) -> np.ndarray:

        old_shape = [*uv_left.shape[:-1], 3]
        uv_left, uv_right = uv_left.reshape(-1, 2), uv_right.reshape(-1, 2)

        # uv_left = self.left_cam.undistort_points(uv_left)
        # uv_right = self.right_cam.undistort_points(uv_right)

        xyz_h = cv2.triangulatePoints(
            self.left_cam.P,
            self.right_cam.P,
            uv_left.T,
            uv_right.T,
        )
        xyz_h = xyz_h.T
        xyz = xyz_h[..., :-1] / xyz_h[..., -1:]
        return np.reshape(xyz, old_shape)


@dataclass
class ImageData:
    image: np.ndarray
    rgb: np.ndarray
    filename: str
    keypoints = None
    features = None
    clahe_img: np.ndarray = None

    def get_features(self):
        self.clahe_img, self.keypoints, self.features = SIFT(self.image)
        return self

    def clear(self):
        self.keypoints = None
        self.features = None


@dataclass
class MatchInfo:
    src_pts_pre_match: np.ndarray
    dst_pts_pre_match: np.ndarray
    src_idx_match: np.ndarray
    dst_idx_match: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray


@dataclass
class ImageMatches:
    image_a: ImageData
    image_b: ImageData
    idx_1: int
    idx_2: int
    match_info: MatchInfo = None
    pts_3D: np.ndarray = None
    inliers: np.ndarray[bool] = None
    E: np.ndarray = None
    filtered_inliers: dict[str, np.ndarray] = field(default_factory=lambda: {})

    @property
    def src_pts(self):
        return self.match_info.src_pts

    @property
    def dst_pts(self):
        return self.match_info.dst_pts

    @property
    def final_inliers(self):
        """Inliers that span [idx_{i-1}, idx_i] and [idx_i, idx_{i+1}]"""

        if not len(self.filtered_inliers):
            return self.inliers
        elif len(self.filtered_inliers) == 1:
            if "curr" in self.filtered_inliers:
                return self.filtered_inliers["curr"]
            else:
                return self.filtered_inliers["prev"]
        else:
            return self.filtered_inliers["curr"]

    def update_stereo_cam(self, cam_1_global_pose: np.ndarray, cam_2_global_pose: np.ndarray):
        self.stereo_cam = StereoCamera(
            R_0=cam_1_global_pose[:3, :3],
            t_0=cam_1_global_pose[:3, -1],
            R_1=cam_2_global_pose[:3, :3],
            t_1=cam_2_global_pose[:3, -1],
        )
        self.pts_3D = self.stereo_cam.triangulate(self.src_pts, self.dst_pts)


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

    # U, _, Vt = np.linalg.svd(E)
    # Ess = U @ np.diag([1,1,0]) @ Vt
    # U, _, Vt = np.linalg.svd(Ess)

    # W = np.matrix('0 -1 0; 1 0 0; 0 0 1')
    # rotations = [U @ W @ Vt, U @ W @ Vt, U @ W.T @ Vt, U @ W.T @ Vt]
    # translations = [U[:, 2], -U[:, 2], U[:, 2], -U[:, 2]] # Columns

    # modified_rotations = []
    # modified_translations = []

    # for R, C in zip(rotations, translations):
    #     C_reshaped = C.reshape(-1, 1)
    #     if np.linalg.det(R) < 0:
    #         R = -1 * R
    #         C_reshaped = -1 * C_reshaped
    #     modified_rotations.append(R)
    #     modified_translations.append(C_reshaped)

    # best_index =[]
    # P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Initial projection matrix

    # for R, C in zip(modified_rotations, modified_translations):
    #     P2 = K @ np.hstack((R, C))
    #     pts4d = cv2.triangulatePoints(P1, P2, src_pts.squeeze().T, dst_pts.squeeze().T) # [4, N]
    #     pts4d /= pts4d[3, :]  # Normalize
    #     pts3d = pts4d[:3]
    #     pts3d = pts3d.T # [N, 3]

    #     valid_idx = (pts4d[2, :] > 0) & \
    #             (pts4d[0, :] > -160) & (pts4d[0, :] < 160) & \
    #             (pts4d[1, :] > -200) & (pts4d[1, :] < 200) & \
    #             (pts4d[2, :] > -20) & (pts4d[2, :] < 500)
    #     pts3d = pts3d[valid_idx]

    #     R_3_T = R.T @ np.array([[0, 0, 1]]).T
    #     view_direction = (pts3d - C.reshape(1, 3)) # [N, 3]
    #     Z_values = view_direction @ R_3_T
    #     positive_depth = np.sum(Z_values > 0)
    #     best_index.append(positive_depth)

    # finalindex = np.argmax(best_index)
    # R_new = modified_rotations[finalindex]
    # t_new = modified_translations[finalindex]


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


def compute_relative_pose(
    image_1: ImageData, image_2: ImageData
) -> tuple[MatchInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the pose from image_1 to image_2 (T_21). The resulting rotation
    and translation are [R | t] from camera 1 to camera 2.

    If you want to get camera 2's center and orientation in camera 1's
    coordinate system (T_12), you need `R_new = -inv(R) @ t` and `t_new = R.T @ t`

    Parameters
    ----------
    image_1 : ImageData
        Image 1
    image_2 : ImageData
        Image 2

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        _description_
    """

    src_pts_pre_match, dst_pts_pre_match, src_idx_match, dst_idx_match, src_pts, dst_pts = BFMatcher(
        image_1.keypoints, image_1.features, image_2.keypoints, image_2.features, CONFIG.threshold
    )

    match_info = MatchInfo(
        src_pts_pre_match=src_pts_pre_match,
        dst_pts_pre_match=dst_pts_pre_match,
        src_idx_match=src_idx_match,
        dst_idx_match=dst_idx_match,
        src_pts=src_pts,
        dst_pts=dst_pts,
    )

    if src_pts.shape[0] == 0 or dst_pts.shape[0] == 0:
        return src_pts, dst_pts, None, None, None, None

    E, mask = cv2.findEssentialMat(
        src_pts,
        dst_pts,
        K,
        method=cv2.RANSAC,
        prob=CONFIG.confidence,
        threshold=CONFIG.ransacReprojThreshold,
    )

    if E is None:
        return src_pts, dst_pts, E, None, None, None

    if E.shape[0] > 3 and E.shape[0] % 3 == 0:
        R_out, t_out, mask_out = find_optimal_E(E, src_pts, dst_pts, K, mask)
    else:
        _, R_out, t_out, mask_out = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)

    return match_info, E, mask_out, R_out, t_out


def compute_triangulation_angle(pose_1: np.ndarray, pose_2: np.ndarray, point3D) -> float:

    ray1 = np.asanyarray(point3D) - np.array(pose_1[:3, -1])
    ray2 = np.asanyarray(point3D) - np.array(pose_2[:3, -1])

    ray1_norm = ray1 / np.linalg.norm(ray1)
    ray2_norm = ray2 / np.linalg.norm(ray2)

    # Calculate the angle using dot product
    cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)  # Angle in radians
    return np.degrees(angle)  # Convert to degrees


def gather(main_array: np.ndarray, sub_array: np.ndarray) -> np.ndarray:
    """
    Gather the indicies where the sub_array comes from within
    the main_array.

    Example:
    - main_array = np.array([5, 4, 3, 2, 1])
    - sub_array = np.array([1, 4, 2])
    indices where sub_array comes from within main_array - [4, 1, 3]

    Parameters
    ----------
    main_array : np.ndarray
        Main array with original values
    sub_array : np.ndarray
        Sub array with a subset of the values

    Returns
    -------
    np.ndarray
        Indices where the values in sub array comes from within the main array.
    """

    sorted_indices = np.argsort(main_array)
    sorted_main_array = main_array[sorted_indices]
    sorted_positions = np.searchsorted(sorted_main_array, sub_array)
    return sorted_indices[sorted_positions]


def get_intersecting_inliers(
    prev_reference: ImageMatches, curr_reference: ImageMatches, prev_inliers: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Between matches [image_i, image_j] and [image_j, image_k], will get
    the inliers in image_j shared in both instances. Will then filter the
    respective inliers between [image_i, image_j] and [image_j, image_k].

    Example
    -------
    `prev_reference`
    dst_idx_match:      [0,2,4,6,9,8]
    inliers:            [1,1,0,0,1,0]
    prev_idx_inliers:   [0,2,    9]

    `curr_reference`
    src_idx_match:      [1,3,2,9,7]
    inliers:            [1,0,0,1,1]
    curr_idx_inliers:   [1,    9,7]

    common_idx_inliers: [9]
    prev_inliers_filt:  [0,0,0,0,1,0]
    curr_inliers_filt:  [0,0,0,1,0]

    Parameters
    ----------
    prev_reference : ImageMatches
        Match between image_i and image_j
    curr_reference : ImageMatches
        Match between image_j and image_k

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        _description_
    """

    if prev_inliers is None:
        prev_inliers = prev_reference.inliers

    prev_idx_inliers: np.ndarray[int] = prev_reference.match_info.dst_idx_match[prev_inliers]
    curr_idx_inliers: np.ndarray[int] = curr_reference.match_info.src_idx_match[curr_reference.inliers]
    common_idx_inliers = np.intersect1d(prev_idx_inliers, curr_idx_inliers)

    prev_inliers_filt = np.zeros_like(prev_reference.inliers).astype(bool)
    _idx = gather(main_array=prev_reference.match_info.dst_idx_match, sub_array=common_idx_inliers)
    prev_inliers_filt[_idx] = True

    curr_inliers_filt = np.zeros_like(curr_reference.inliers).astype(bool)
    _idx = gather(main_array=curr_reference.match_info.src_idx_match, sub_array=common_idx_inliers)
    curr_inliers_filt[_idx] = True

    return common_idx_inliers, prev_inliers_filt, curr_inliers_filt


def pose_to_matrix(pose):
    return np.array(pose.matrix())


def visualize_graph(
    estimates,
    colors: dict[gtsam.Symbol, np.ndarray],
    pt_scale: int = 3,
    title: str = "SFM + Trajectory",
    debug_key: str = "",
):
    import gtsam.utils.plot

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

    # gtsam.utils.plot.plot_3d_points(1, sfm.initial_estimate)
    gtsam.utils.plot.plot_trajectory(1, estimates, scale=3, title=title)
    gtsam.utils.plot.set_axes_equal(1)

    ax = plt.gca()
    ax.view_init(elev=-75, azim=-90)

    save_pcd(landmarks, display_colors, camera_frames, title=title)
    plt.show()


def save_pcd(
    landmarks: np.ndarray, display_colors: np.ndarray, camera_frames: list[np.ndarray], title: str = "SFM + Trajectory"
):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(landmarks)
    point_cloud.colors = o3d.utility.Vector3dVector(display_colors)

    output_filename = "_".join(title.split())
    output_filename = f"{output_filename}.ply"
    o3d.io.write_point_cloud(os.path.join(__cwd__, output_filename), point_cloud)

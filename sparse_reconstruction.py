import gtsam
import numpy as np
from gtsam.symbol_shorthand import C as Camera
from gtsam.symbol_shorthand import L as Landmark
from scipy.spatial.transform import Rotation as Rotation

from config import K
from utils import ImageMatches


class SparseReconstruction:
    def __init__(self, local_poses: dict[tuple[int, int], dict[str, np.ndarray]]):

        self.K = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        self.point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.camera_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        # self.camera_noise = gtsam.noiseModel.Robust.Create(
        #     gtsam.noiseModel.mEstimator.Huber(5.0), gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        # )

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])  # 30 cm and 0.1 rad
        )
        self.local_poses = local_poses
        self.colors = {}
        self.initialized = False

    def add_prior_pose3_factor(self, key: int):

        self.graph.add(
            gtsam.PriorFactorPose3(
                key,
                gtsam.Pose3(),
                # gtsam.noiseModel.Diagonal.Variances([10, 10, 10, 10, 10, 10]),
                self.pose_noise,
            )
        )

    def add_prior_point_factor(self, key: int, pt_3d: np.ndarray):

        self.graph.add(
            gtsam.PriorFactorPoint3(
                key,
                gtsam.Point3(*pt_3d),
                self.point_noise,
            )
        )

    def add_between_poses(self, idx_1: int, idx_2: int):

        R = self.local_poses[(idx_1, idx_2)]["R"]
        t = self.local_poses[(idx_1, idx_2)]["t"]

        relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t.flatten()))

        factor = gtsam.BetweenFactorPose3(Camera(idx_1), Camera(idx_2), relative_pose, self.pose_noise)
        self.graph.add(factor)

    def add_initial_cam_pose_estimate(self, global_poses: dict[int, np.ndarray], idx: int):

        global_pose = global_poses[idx]
        R, t = global_pose[:3, :3], global_pose[:3, -1]

        global_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t.flatten()))
        self.initial_estimate.insert(Camera(idx), global_pose)

    def _add_projection_factor(self, pt: np.ndarray, cam_idx: int, landmark_idx: int):

        factor = gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(*pt), self.camera_noise, Camera(cam_idx), Landmark(landmark_idx), self.K
        )
        self.graph.add(factor)

    def add_projection_factors(
        self, src_pts: np.ndarray, dst_pts: np.ndarray, cam_0: int, cam_1: int, landmark_idx: int
    ) -> list[int]:

        valid_landmarks_indices = []
        for pt_idx, (src, dst) in enumerate(zip(src_pts, dst_pts)):
            valid_landmarks_indices.append(pt_idx)
            self._add_projection_factor(src, cam_0, landmark_idx + pt_idx)
            self._add_projection_factor(dst, cam_1, landmark_idx + pt_idx)

        return valid_landmarks_indices

    def solve(self):

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, result)
        return result, marginals

    def is_valid(self, idx_1: int, idx_2: int) -> bool:

        pose_dict = self.local_poses[(idx_1, idx_2)]
        relative_pose = gtsam.Pose3(gtsam.Rot3(pose_dict["R"]), gtsam.Point3(pose_dict["t"].flatten()))

        t_norm = np.linalg.norm(relative_pose.translation())
        r_angle = np.linalg.norm(relative_pose.rotation().rpy())
        return np.rad2deg(r_angle)

    def add(
        self,
        global_poses: dict[int, np.ndarray],
        image_matches: list[ImageMatches],
        use_shared_inliers: bool = False,
        init_landmark_idx: int = 0,
        debug: bool = False,
    ):
        landmark_idx = init_landmark_idx

        for i, matches in enumerate(image_matches):
            # print("LANDMARK IDX", landmark_idx)

            rad = self.is_valid(matches.idx_1, matches.idx_2)
            if debug:
                print(f"For pose <{matches.idx_1, matches.idx_2}>, angle is {rad} degrees.")

            # if not rad > 2:
            #     print("---" * 10)
            #     continue

            msg = (
                f"For pose <{matches.idx_1, matches.idx_2}>, there are "
                f"{len(matches.pts_3D)} matches and {sum(matches.inliers)} "
                f"inliers and {sum(matches.final_inliers)} final inliers."
            )
            if debug:
                print(msg)

            if use_shared_inliers:
                inliers = matches.final_inliers
            else:
                inliers = matches.inliers
            pts_3D = matches.pts_3D.squeeze()[inliers]
            src = matches.src_pts.squeeze()[inliers]
            dst = matches.dst_pts.squeeze()[inliers]
            assert pts_3D.shape[0] == src.shape[0] == dst.shape[0]

            self.add_between_poses(matches.idx_1, matches.idx_2)
            if not self.initial_estimate.exists(Camera(matches.idx_1)):
                self.add_initial_cam_pose_estimate(global_poses, idx=matches.idx_1)
            if not self.initial_estimate.exists(Camera(matches.idx_2)):
                self.add_initial_cam_pose_estimate(global_poses, idx=matches.idx_2)

            valid_landmarks_indices = self.add_projection_factors(
                src_pts=src, dst_pts=dst, cam_0=matches.idx_1, cam_1=matches.idx_2, landmark_idx=landmark_idx
            )

            if not self.initialized and len(valid_landmarks_indices) > 0:
                prior_index = valid_landmarks_indices[0]
                for prior_index in valid_landmarks_indices:
                    self.add_prior_point_factor(Landmark(prior_index), pts_3D[prior_index])
                self.initialized = True

            if self.initialized:
                for l in valid_landmarks_indices:
                    self.initial_estimate.insert(Landmark(l + landmark_idx), gtsam.Point3(*pts_3D[l]))
                    x, y = src[l].astype(int)
                    self.colors[Landmark(l + landmark_idx)] = matches.image_a.rgb[y, x]

            landmark_idx += len(valid_landmarks_indices)
            if debug:
                print("---" * 10)

import itertools
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gtsam
import networkx as nx
import numpy as np
from gtsam.symbol_shorthand import C as Camera
from gtsam.symbol_shorthand import L as Landmark
from scipy.spatial.transform import Rotation as Rotation

__cwd__: Path = Path(os.path.dirname(__file__))
sys.path.insert(0, __cwd__)
sys.path = list(set(sys.path))

from utils import (
    ImageData,
    ImageMatches,
    K,
    MatchInfo,
    compute_relative_pose,
    gather,
    get_intersecting_inliers,
)


class Matcher:
    def __init__(self, images: list[ImageData]):

        self.images = images
        self.executor = ThreadPoolExecutor()
        self.H, self.W = images[0].image.shape[:2]

        _exec = ThreadPoolExecutor()

        def get_features(image: ImageData):
            image.get_features()

        futures = []
        for image in self.images:
            futures.append(_exec.submit(get_features, image))
        [img for img in (future.result() for future in as_completed(futures))]
        _exec.shutdown()

        # Contains mapping of (key1, key2) to R, t, and transformation.
        # R, t and T are in format from key2 to key1.
        # T_(key1,key2)
        self.local_poses: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self.graph = nx.Graph()

    def _get_pair_pose(self, i: int, j: int, record_global: bool) -> ImageMatches:

        image_i = self.images[i]
        image_j = self.images[j]
        match_info, E, mask, R_ji, t_ji = compute_relative_pose(image_i, image_j)  # to image_j from image_i

        if record_global:
            assert E is not None, f"Modify CONFIG to get more features for {(i, j)}!"

            T_ji = np.eye(4)
            T_ji[:3, :3] = R_ji
            T_ji[:3, -1] = t_ji.squeeze()
            self.local_poses[(j, i)] = {"R": R_ji, "t": t_ji, "T": T_ji}

            T_ij = np.linalg.inv(T_ji)
            self.local_poses[(i, j)] = {"R": T_ij[:3, :3], "t": T_ij[:3, -1], "T": T_ij}

        inliers = mask.ravel().astype(bool)
        if E is None or sum(inliers) < 10:
            return None

        pair_match = ImageMatches(
            image_a=image_i, image_b=image_j, idx_1=i, idx_2=j, match_info=match_info, inliers=inliers, E=E
        )

        return pair_match

    def _get_poses(self, possible_pairs: list[tuple[int, int]], record_poses: bool) -> list[ImageMatches]:

        pair_matches = []
        futures = []
        for i, j in possible_pairs:
            futures.append(self.executor.submit(self._get_pair_pose, i, j, record_poses))

        pair_matches: list[ImageMatches] = pair_matches + [
            pair_match
            for pair_match in (future.result() for future in as_completed(fs=futures))
            if pair_match is not None
        ]
        pair_matches.sort(key=lambda x: x.idx_1)
        return pair_matches

    def get_poses(self, possible_pairs: list[tuple[int, int]] | None = None) -> list[ImageMatches]:

        if possible_pairs is None:
            possible_pairs = list(zip(range(0, len(self.images) - 1), range(1, len(self.images))))
        sequential_matches = self._get_poses(possible_pairs, record_poses=True)

        non_squential = list(itertools.permutations(iterable=range(len(self.images)), r=2))
        non_squential = list(filter(lambda x: x[1] - 1 != x[0], non_squential))

        print("Done getting matches...")
        return sequential_matches

    def __len__(self):
        return len(self.images)

    def __del__(self):
        self.executor.shutdown()


class FixScaleAmbiguity:

    def __init__(
        self, sequential_matches: list[ImageMatches], local_poses: dict[tuple[int, int], dict[str, np.ndarray]]
    ):

        self.global_poses_ji = {
            sequential_matches[0].idx_1: np.eye(4),
            sequential_matches[0].idx_2: local_poses[(sequential_matches[0].idx_2, sequential_matches[0].idx_1)]["T"],
        }

        self.global_poses_ij = {
            sequential_matches[0].idx_1: np.eye(4),
            sequential_matches[0].idx_2: local_poses[(sequential_matches[0].idx_1, sequential_matches[0].idx_2)]["T"],
        }

        self.local_poses = local_poses
        self.new_local_poses = dict(local_poses)
        self.sequential_matches = sequential_matches

        # Set base match as match whose translation will be the ground-truth for scale
        self.base_idx = 0
        self.base_match = sequential_matches[self.base_idx]

        match_info_inv = MatchInfo(
            src_pts_pre_match=self.base_match.match_info.dst_pts_pre_match,
            dst_pts_pre_match=self.base_match.match_info.src_pts_pre_match,
            src_idx_match=self.base_match.match_info.dst_idx_match,
            dst_idx_match=self.base_match.match_info.src_idx_match,
            src_pts=self.base_match.match_info.dst_pts,
            dst_pts=self.base_match.match_info.src_pts,
        )

        self.base_match_inv = ImageMatches(
            image_a=self.base_match.image_b,
            image_b=self.base_match.image_a,
            idx_1=self.base_match.idx_2,
            idx_2=self.base_match.idx_1,
            match_info=match_info_inv,
            pts_3D=self.base_match.pts_3D,
            inliers=self.base_match.inliers,
            E=None,
        )

    def _update_local_poses(self, a: int, b: int, T: np.ndarray):

        self.new_local_poses[(a, b)]["T"] = T
        self.new_local_poses[(a, b)]["R"] = T[:3, :3]
        self.new_local_poses[(a, b)]["t"] = T[:3, -1]

    def _retriangulate_match(self, match: ImageMatches, common_inliers: np.ndarray) -> np.ndarray:

        src_pts = match.src_pts[common_inliers]
        dst_pts = match.dst_pts[common_inliers]

        # Get 3D points
        assert src_pts.shape == dst_pts.shape
        reference_pts_3D = match.stereo_cam.triangulate(src_pts, dst_pts)
        return reference_pts_3D

    def _get_scale(
        self,
        prev_reference: ImageMatches,
        prev_inliers_filt: np.ndarray,
        curr_reference: ImageMatches,
        curr_inliers_filt: np.ndarray,
    ) -> float:

        reference_pts_3D = prev_reference.pts_3D[prev_inliers_filt]
        current_pts_3D = curr_reference.pts_3D[curr_inliers_filt]

        # Fix scale ambiguity
        ref_pts_norm: np.ndarray = np.linalg.norm(reference_pts_3D, axis=-1)
        cur_pts_norm: np.ndarray = np.linalg.norm(current_pts_3D, axis=-1)
        scale = ref_pts_norm.mean() / cur_pts_norm.mean()
        print("scale: ", scale)
        return scale

    def _fix_and_update_scale(
        self, prev_reference: ImageMatches, curr_reference: ImageMatches
    ) -> tuple[np.ndarray, np.ndarray]:

        # Find overlapping inlier's indices in both instances of image_j
        common_idx_inliers, prev_inliers_filt, curr_inliers_filt = get_intersecting_inliers(
            prev_reference, curr_reference
        )

        if len(common_idx_inliers) > 2:
            scale = self._get_scale(prev_reference, prev_inliers_filt, curr_reference, curr_inliers_filt)
        else:
            scale = 1.0

        T_ji = self.local_poses[(curr_reference.idx_2, curr_reference.idx_1)]["T"]
        T_ji[:3, -1] = T_ji[:3, -1] * scale
        T_ij = np.linalg.inv(T_ji)

        self.global_poses_ij[curr_reference.idx_2] = T_ij.dot(self.global_poses_ij[curr_reference.idx_1])
        self.global_poses_ji[curr_reference.idx_2] = T_ji.dot(self.global_poses_ji[curr_reference.idx_1])

        self._update_local_poses(curr_reference.idx_1, curr_reference.idx_2, T_ij)
        self._update_local_poses(curr_reference.idx_2, curr_reference.idx_1, T_ji)

        curr_reference.update_stereo_cam(
            cam_1_global_pose=self.global_poses_ji[curr_reference.idx_1],
            cam_2_global_pose=self.global_poses_ji[curr_reference.idx_2],
        )
        return prev_inliers_filt, curr_inliers_filt

    def fix_scale_ambiguity_sequential(self):
        """
        Updates self.sequential_matches by rescaling the stereo camera
        between 2 matches.
        """

        for i in range(len(self.sequential_matches)):
            if i == self.base_idx:
                continue

            prev_reference = self.sequential_matches[i - 1]  # [image_{i-1}, image_i]
            curr_reference = self.sequential_matches[i]  # [image_i, image_{i+1}]

            prev_inliers_filt, curr_inliers_filt = self._fix_and_update_scale(prev_reference, curr_reference)
            self.sequential_matches[i].filtered_inliers = {
                "prev": prev_inliers_filt,
                "curr": curr_inliers_filt,
            }

        self.sequential_matches[self.base_idx].filtered_inliers = {
            "curr": self.sequential_matches[self.base_idx + 1].filtered_inliers["prev"],
        }

    def fix_scale_ambiguity_nonsequential(self, loop_closure_matches: list[ImageMatches]):
        """
        Updates loop_closure_matches by rescaling the stereo camera
        between 2 matches.
        """

        idx1_to_position: dict[int, int] = {}
        for idx, m in enumerate(self.sequential_matches):
            idx1_to_position[m.idx_1] = idx

        for loop_match in loop_closure_matches:
            position = idx1_to_position[loop_match.idx_1]

            if position == self.base_idx:
                # For this case in loop closure we have:
                # loop_match: [image_0, image_n]
                # However, since there is no image_{-1}, we can use
                # [image_1, image_0] as the "base", i.e, self.base_match_inv
                prev_reference = self.base_match_inv
            else:
                prev_reference = self.sequential_matches[position - 1]

            assert prev_reference.idx_2 == loop_match.idx_1

            prev_inliers_filt, curr_inliers_filt = self._fix_and_update_scale(prev_reference, loop_match)
            loop_match.filtered_inliers = {
                "prev": prev_inliers_filt,
                "curr": curr_inliers_filt,
            }


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


class FindCommonInliers:

    def __init__(self, matches: list[ImageMatches]):

        self.loop_closure_candidates = defaultdict(lambda: [])
        self.total_inliers = defaultdict(lambda: [])
        self.matches = matches

    def _add_to_totals(
        self,
        inliers_idx: np.ndarray,
        base_landmark_idx: int,
        idx_1: int,
        idx_2: int,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        pts_3D: np.ndarray,
    ):
        assert len(inliers_idx) == len(src_pts) == len(dst_pts) == len(pts_3D)
        for idx, src, dst, pt_3D in zip(inliers_idx, src_pts, dst_pts, pts_3D):
            key = idx + base_landmark_idx
            self.total_inliers[key].append([idx_1, idx_2, src, dst, pt_3D])

    def get_inliers(self):

        # src_pts -> queryIdx
        # dst_pts -> trainIdx

        landmark_idx = 0
        for i, prev_match in enumerate(self.matches):
            idx_1 = prev_match.idx_1
            idx_2 = prev_match.idx_2

            # print(">>>>> LANDMARK IDX", landmark_idx)
            print(f"Match: {idx_1, idx_2}, has inliers that will propagate up to ...", end="\r")

            original_dst_idx_match = prev_match.match_info.dst_idx_match
            num_inliers = sum(prev_match.inliers)
            cummulative = 0
            new_inliers_mask = prev_match.inliers

            assert len(prev_match.pts_3D) == len(prev_match.inliers)
            self._add_to_totals(
                inliers_idx=range(sum(prev_match.inliers)),
                base_landmark_idx=landmark_idx,
                idx_1=idx_1,
                idx_2=idx_2,
                src_pts=prev_match.src_pts[prev_match.inliers],
                dst_pts=prev_match.dst_pts[prev_match.inliers],
                pts_3D=prev_match.pts_3D[prev_match.inliers],
            )

            backward_links = []
            original_src_pts = prev_match.src_pts

            for j in range(i + 1, len(self.matches)):
                new_match = self.matches[j]

                trainIdx_inliers: np.ndarray[int] = prev_match.match_info.dst_idx_match[new_inliers_mask]
                queryIdx_inliers: np.ndarray[int] = new_match.match_info.src_idx_match[new_match.inliers]
                forward_propagated_inliers_idx = np.intersect1d(trainIdx_inliers, queryIdx_inliers)

                new_inliers_mask = np.zeros_like(new_match.inliers).astype(bool)
                _idx = gather(main_array=new_match.match_info.src_idx_match, sub_array=forward_propagated_inliers_idx)
                new_inliers_mask[_idx] = True

                src_idxs = new_match.match_info.src_idx_match[new_inliers_mask]
                dst_idxs = new_match.match_info.dst_idx_match[new_inliers_mask]
                if cummulative == 0:
                    original_inliers_mask = gather(
                        main_array=original_dst_idx_match, sub_array=forward_propagated_inliers_idx
                    )
                    backward_links.insert(0, dict(zip(forward_propagated_inliers_idx, original_inliers_mask)))
                else:
                    original_inliers_mask = []
                    for propagated_idx in forward_propagated_inliers_idx:
                        for back_links in backward_links:
                            propagated_idx = back_links[propagated_idx]
                        original_inliers_mask.append(propagated_idx)
                    original_inliers_mask = np.array(original_inliers_mask)
                backward_links.insert(0, dict(zip(dst_idxs, src_idxs)))

                # print("forward propagated in", forward_propagated_inliers_idx)
                # print("original idx", original_idx)

                if sum(new_inliers_mask) == 0:
                    prev_match = new_match
                    break

                if sum(new_inliers_mask) >= 5:
                    self.loop_closure_candidates[idx_1].append(new_match.idx_2)
                cummulative += sum(prev_match.inliers)
                print(
                    f"Match: {idx_1, idx_2}, has inliers that will propagate up to ... {new_match.idx_1, new_match.idx_2}",
                    end="\r",
                )
                assert len(new_match.pts_3D) == len(new_inliers_mask)
                inliers_idx = gather(np.flatnonzero(new_match.inliers), np.flatnonzero(new_inliers_mask))

                self._add_to_totals(
                    inliers_idx=inliers_idx,
                    base_landmark_idx=landmark_idx + cummulative,
                    idx_1=idx_1,
                    idx_2=new_match.idx_2,
                    src_pts=original_src_pts[original_inliers_mask],
                    dst_pts=new_match.dst_pts[new_inliers_mask],
                    pts_3D=new_match.pts_3D[new_inliers_mask],
                )
                prev_match = new_match

            print("\n", end="\r")
            landmark_idx += num_inliers


# DEBUG CODE FOR FindCommonInliers
########## ============================================================================================================ ###########

# landmark_idx = 0
# example_matches = {
#     (0, 1): {
#         'queryIdx': np.array([7, 5, 8, 11, 2, 0, 6, 9]),
#         'trainIdx': np.array([1, 4, 7, 3, 13, 5, 2, 6]),
#         'inliers': np.array([1, 1, 0, 1, 1, 1, 1, 0]).astype(bool)
#     },
#     (1, 2): {
#         'queryIdx': np.array([3, 0, 2, 12, 6, 15, 5, 20]),
#         'trainIdx': np.array([5, 11, 3, 17, 0, 2, 6, 20]),
#         'inliers': np.array([1, 0, 0, 1, 1, 0, 1, 0]).astype(bool)
#     },
#     (2, 3): {
#         'queryIdx': np.array([1, 3, 5, 9, 4, 13, 12, 0]),
#         'trainIdx': np.array([7, 8, 10, 4, 5, 1, 2, 0]),
#         'inliers': np.array([0, 0, 1, 1, 0, 1, 1, 0]).astype(bool)
#     }
# }

# for (idx_1, idx_2), info in example_matches.items():
#     print("*************************** LANDMARK IDX", landmark_idx, "***************************")

#     dst_idx_match = info["trainIdx"]
#     num_inliers = sum(info["inliers"])
#     cummulative = 0
#     new_inliers_mask = info["inliers"]
#     print(new_inliers_mask)
#     backward_links = []
#     for j in range(idx_2, len(example_matches)):
#         print("*********", j, j+1, "*********")
#         new_info = example_matches[(j, j+1)]

#         trainIdx_inliers: np.ndarray[int] = info["trainIdx"][new_inliers_mask]
#         queryIdx_inliers: np.ndarray[int] = new_info["queryIdx"][new_info["inliers"]]
#         print("train idx", trainIdx_inliers)
#         print("query idx", queryIdx_inliers)
#         forward_propagated_inliers_idx = np.intersect1d(trainIdx_inliers, queryIdx_inliers)

#         new_inliers_mask = np.zeros_like(new_info["inliers"]).astype(bool)
#         _idx = gather(main_array=new_info["queryIdx"], sub_array=forward_propagated_inliers_idx)
#         new_inliers_mask[_idx] = True

#         src_idxs = new_info["queryIdx"][new_inliers_mask]
#         dst_idxs = new_info["trainIdx"][new_inliers_mask]
#         if cummulative == 0:
#             original_idx = gather(main_array=dst_idx_match, sub_array=forward_propagated_inliers_idx)
#             backward_links.insert(0, dict(zip(forward_propagated_inliers_idx, original_idx)))
#         else:
#             original_idx = []
#             for propagated_idx in forward_propagated_inliers_idx:
#                 for back_links in backward_links:
#                     propagated_idx = back_links[propagated_idx]
#                 original_idx.append(propagated_idx)
#             original_idx = np.array(original_idx)
#         backward_links.insert(0, dict(zip(dst_idxs, src_idxs)))

#         if sum(new_inliers_mask) == 0:
#             break
#         cummulative += sum(info["inliers"]) # sum(new_info["inliers"])

#         print("forward propagated idx", forward_propagated_inliers_idx)
#         print("new inliers mask", new_inliers_mask)
#         print("src_idxs", src_idxs)
#         print("dst_idxs", dst_idxs)

#         print("backward links", backward_links)
#         print("original idx", original_idx)

#         inliers_idx = gather(np.flatnonzero(new_info["inliers"]), np.flatnonzero(new_inliers_mask))
#         keys = landmark_idx + cummulative + inliers_idx
#         print("KEYS: ", "inliers idx - ", inliers_idx, "final key - ", keys)
#         info = new_info

#     landmark_idx += num_inliers

########## ============================================================================================================ ###########

from collections import defaultdict

import gtsam
import networkx as nx
import numpy as np
from gtsam.symbol_shorthand import C as Camera
from gtsam.symbol_shorthand import L as Landmark
from scipy.spatial.transform import Rotation as Rotation

from sfm.config import K
from sfm.covisibility import QueryTrainTracks
from sfm.utils import StereoCamera
from sfm.visualization import visualize_graph

point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
camera_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))


class SceneGraph:
    def __init__(self):

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        self.K = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])  # 30 cm and 0.1 rad
        )
        self.point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.camera_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(5.0), gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        )
        self.camera_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        self.initialized = False

    def add_initial_cam_pose_estimate(self, global_pose: np.ndarray, idx: int):
        """
        Add initial estimate of the pose for the given camera index.

        Parameters
        ----------
        global_pose : np.ndarray
            Initial pose estimate as a 4x4 matrix.
        idx : int
            Camera index.
        """

        global_pose = np.linalg.inv(global_pose)
        global_pose = gtsam.Pose3(gtsam.Rot3(global_pose[:3, :3]), gtsam.Point3(global_pose[:3, -1].flatten()))
        if not self.initial_estimate.exists(Camera(idx)):
            self.initial_estimate.insert(Camera(idx), global_pose)

    def add_initial_pt_3D_landmark_estimate(self, id: int, pt_3D: np.ndarray):
        """
        Add initial estimate of the 3D position for the given landmark.

        Parameters
        ----------
        id : int
            Landmark ID.
        pt_3D : np.ndarray
            Initial estimate of the 3D location.
        """

        if not self.initial_estimate.exists(Landmark(id)):
            self.initial_estimate.insert(Landmark(id), gtsam.Point3(*pt_3D))

    def add_projection_factors(self, pt: np.ndarray, cam_idx: int, landmark_idx: int):
        """
        Add projection factor to the graph. Provides information of the 2D location of the
        specified landmark on the specified camera.

        Parameters
        ----------
        pt : np.ndarray
            2D location (or projection) of the landmark on the camera.
        cam_idx : int
            Camera index.
        landmark_idx : int
            Landmark ID.
        """

        factor = gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(*pt),
            self.camera_noise,
            Camera(cam_idx),
            Landmark(landmark_idx),
            self.K,
        )
        self.graph.add(factor)

    def add_prior_point_factor(self, landmark_idx: int, pt_3d: np.ndarray):
        """
        Add PriorFactorPoint3 factor to the graph. Provide a soft constraint on the
        absolute 3D position (Point3) of the specified Landmark.

        Parameters
        ----------
        key : int
            Landmark ID.
        pt_3d : np.ndarray
            3D location.
        """
        self.graph.add(
            gtsam.PriorFactorPoint3(
                Landmark(landmark_idx),
                gtsam.Point3(*pt_3d),
                self.point_noise,
            )
        )

    def add_between_poses(self, idx_1: int, idx_2: int, relative_pose: np.ndarray):
        """
        Adds a BetweenFactorPose3 to the graph to add a soft contraint on the relative pose between
        2 cameras.

        Parameters
        ----------
        idx_1 : int
            Camera index 1.
        idx_2 : int
            Camera index 2.
        relative_pose : np.ndarray
            Relative pose between the 2 cameras.
        """

        relative_pose = gtsam.Pose3(gtsam.Rot3(relative_pose[:3, :3]), gtsam.Point3(relative_pose[:3, -1].flatten()))
        factor = gtsam.BetweenFactorPose3(Camera(idx_1), Camera(idx_2), relative_pose, self.pose_noise)
        self.graph.add(factor)

    def update_graph(
        self,
        G_covisibility: nx.Graph,
        reference_edge: tuple[int, int],
        new_edge: tuple[int, int],
        query_train_tracks: QueryTrainTracks,
        n: int,
    ):
        """
        Update the graph given the results of pose estimation using PnP, triangulation, and feature matchings.

        Parameters
        ----------
        G_covisibility : nx.Graph
            Covisibility graph containing the poses for each camera.
        reference_edge : tuple[int, int]
            Reference edge for new_edge from covisibility graph.
        new_edge : tuple[int, int]
            New camera pair: (cam_i, cam_j), to evaluate.
        query_train_tracks : QueryTrainTracks
            Dataclass with info on tracks that contain the set of cameras in (*new_edge, *reference_edge).
        n : int
            Minimum number of valid points needed for a track (Landmark) to be considered a valid track to update.
        """

        ref_edge_data = G_covisibility.edges[reference_edge]
        new_edge_data = G_covisibility.edges[new_edge]

        ref_stereo_data: StereoCamera = ref_edge_data["stereo_camera"]
        new_stereo_data: StereoCamera = new_edge_data["stereo_camera"]

        self.add_initial_cam_pose_estimate(new_stereo_data.cam_0.pose, new_edge[0])
        self.add_initial_cam_pose_estimate(new_stereo_data.cam_1.pose, new_edge[1])
        # TODO: Add between poses (relative) for new_edge and non-intersection between ref_edge and new_edge

        # Iterate through the tracks of matched points that pass
        # through the cameras: (ref_i, ref_j) and (new_m, new_n)
        for track_obj in query_train_tracks.track_objs:
            if track_obj.is_valid_track(n=n):
                # Since tracks are updated with results from new_edge, we don't need
                # to iterate through the full track, only the most recent results
                (src, dst), pt_3D, inlier = track_obj.get_edge(new_edge)
                if inlier:
                    self.add_initial_pt_3D_landmark_estimate(track_obj.track_id, pt_3D)
                    self.add_projection_factors(src, new_edge[0], track_obj.track_id)
                    self.add_projection_factors(dst, new_edge[1], track_obj.track_id)
                    self.add_prior_point_factor(track_obj.track_id, pt_3D)

    def solve(self):
        """
        Run Levenberg Marquard optimization to solve the constructed graph given the contraints, relations, and noise
        estimates.

        Returns
        -------
        tuple
            Results, marginals, and the errors after optimization
        """

        visualize_graph(self.initial_estimate, defaultdict(lambda: [255, 255, 255]), title="SFM Initial Estimates")
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setMaxIterations(100)

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, result)

        error = self.graph.error(result)
        visualize_graph(result, defaultdict(lambda: [255, 255, 255]), title="SFM Optimized")
        return result, marginals, error

from collections import defaultdict

import gtsam
import natsort
import networkx as nx
import numpy as np
from gtsam.symbol_shorthand import C as Camera
from gtsam.symbol_shorthand import L as Landmark
from scipy.spatial.transform import Rotation as Rotation

from sfm.config import K
from sfm.covisibility import Track
from sfm.utils import StereoCamera
from sfm.visualization import visualize_graph

point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
camera_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))


def optimize(G_covisibility: nx.DiGraph, valid_tracks: list[Track]):

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    k = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

    camera_attrs = [(camera_0, camera_1, data) for camera_0, camera_1, data in G_covisibility.edges(data=True)]
    camera_attrs = natsort.natsorted(camera_attrs, key=lambda x: x[-1]["num_pts_3D"])

    # Add initial estimate for camera poses
    for camera_0, camera_1, data in camera_attrs:
        if data["num_pts_3D"] > 0:
            stereo_camera: StereoCamera = data["stereo_camera"]
            global_pose_0, global_pose_1 = stereo_camera.get_gtsam_poses()

            if not initial_estimate.exists(Camera(camera_0)):
                initial_estimate.insert(Camera(camera_0), global_pose_0)  # TODO: Are cameras inverted?
            if not initial_estimate.exists(Camera(camera_1)):
                initial_estimate.insert(Camera(camera_1), global_pose_1)

            # TODO: graph.add(gtsam.BetweenFactorPose3(Camera(camera_0), Camera(camera_1), relative_pose, pose_noise))

    # Add initial estimates for each track and projection factors
    for track_obj in valid_tracks:
        edges = track_obj.get_edges()
        for camera_0, camera_1 in edges:
            (src, dst), pt_3D, inlier = track_obj.get_edge((camera_0, camera_1))
            if inlier:
                if not initial_estimate.exists(Landmark(track_obj.track_id)):
                    initial_estimate.insert(Landmark(track_obj.track_id), gtsam.Point3(*pt_3D))

                graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        gtsam.Point2(*src), camera_noise, Camera(camera_0), Landmark(track_obj.track_id), k
                    )
                )
                graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(
                        gtsam.Point2(*dst), camera_noise, Camera(camera_1), Landmark(track_obj.track_id), k
                    )
                )
                graph.add(
                    gtsam.PriorFactorPoint3(
                        Landmark(track_obj.track_id),
                        gtsam.Point3(*pt_3D),
                        point_noise,
                    )
                )

    visualize_graph(initial_estimate, defaultdict(lambda: [255, 255, 255]), title="SFM Initial Estimates")

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(100)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)

    error = graph.error(result)
    visualize_graph(result, defaultdict(lambda: [255, 255, 255]), title="SFM Optimized")
    return result, marginals, error

from collections import defaultdict

import gtsam
import natsort
import networkx as nx
import numpy as np
from gtsam.symbol_shorthand import C as Camera
from gtsam.symbol_shorthand import L as Landmark
from scipy.spatial.transform import Rotation as Rotation

from config import K
from covisibility import Track
from utils import StereoCamera
from visualization import visualize_graph

point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
camera_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))


def setup_graph(G_covisibility: nx.DiGraph, init_tracks: list[Track], valid_tracks: list[Track]):

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    k = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

    # Add prior factors initial camera pair's tracks
    for track_obj in init_tracks:
        edges = track_obj.get_edges()
        for camera_0, camera_1 in edges:
            _, pt_3D, inlier = track_obj.get_edge((camera_0, camera_1))
            if inlier and track_obj.is_valid_track(n=1):
                if not initial_estimate.exists(Landmark(track_obj.track_id)):
                    initial_estimate.insert(Landmark(track_obj.track_id), gtsam.Point3(*pt_3D))
                graph.add(
                    gtsam.PriorFactorPoint3(
                        Landmark(track_obj.track_id),
                        gtsam.Point3(*pt_3D),
                        point_noise,
                    )
                )

    # Add initial estimate for camera poses
    camera_poses = set()
    camera_attrs = [(camera_0, camera_1, data) for camera_0, camera_1, data in G_covisibility.edges(data=True)]
    camera_attrs = natsort.natsorted(camera_attrs, key=lambda x: x[-1]["num_pts_3D"])

    for camera_0, camera_1, data in camera_attrs:
        if data["num_pts_3D"] > 0:

            stereo_camera: StereoCamera = data["stereo_camera"]
            global_pose_0, global_pose_1 = stereo_camera.get_gtsam_poses()

            if camera_0 not in camera_poses:
                initial_estimate.insert(Camera(camera_0), global_pose_0)
            if camera_1 not in camera_poses:
                initial_estimate.insert(Camera(camera_1), global_pose_1)

            camera_poses.update([camera_0, camera_1])
            # graph.add(gtsam.BetweenFactorPose3(Camera(camera_0), Camera(camera_1), relative_pose, pose_noise))

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

    visualize_graph(initial_estimate, defaultdict(lambda: [255, 255, 255]))

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)
    return result, marginals

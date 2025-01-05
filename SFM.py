import argparse
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from sfm.covisibility import (
    add_to_covisibility_graph,
    add_to_tracks_graph,
    get_common_track_IDs,
    get_edge_with_largest_weight,
    get_next_edge,
    get_query_and_train,
    obtrain_tracks,
)
from sfm.extract_features import get_image_loader
from sfm.image_matching import get_image_matcher_loader
from sfm.sparse_reconstruction import setup_graph
from sfm.utils import (
    ImageData,
    StereoCamera,
    filter_inliers,
    get_edge_relation,
    get_pnp_pose,
    get_pose,
    get_src_dst_pts,
    logger,
)
from sfm.visualization import plot_matches

IMG_DIR = Path(__file__).parent / "images"


def preprocess() -> tuple[nx.DiGraph, nx.DiGraph, list[ImageData]]:

    image_data: list[ImageData] = []
    G_tracks = nx.DiGraph()
    G_covisibility = nx.DiGraph()

    img_loader = get_image_loader(IMG_DIR)
    for idx, (img, clahe, kpt, descriptors) in enumerate(tqdm.tqdm(img_loader, desc="Extracting features")):
        image_data.append(
            ImageData(image=img, rgb=img, filename=f"image_{idx}", keypoints=kpt, features=descriptors, clahe_img=clahe)
        )

    pair_loader = get_image_matcher_loader(image_data)
    for idx, [(camera_0, camera_1, matches)] in enumerate(tqdm.tqdm(pair_loader, desc="Matching features")):
        G_tracks = add_to_tracks_graph(G_tracks, camera_0, camera_1, matches)
        G_covisibility = add_to_covisibility_graph(G_covisibility, camera_0, camera_1, matches)

    return G_tracks, G_covisibility, image_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    G_tracks, G_covisibility, image_data = preprocess()
    tracks, camera2trackIDs, node2trackID = obtrain_tracks(G_tracks)

    # Get initial edge
    init_edge = (0, 1)  # get_edge_with_largest_weight(G_covisibility)
    queryIdx_init, trainIdx_init, _, _, track_objs = get_query_and_train(init_edge, tracks, camera2trackIDs)

    # Get corresponding source and destination points
    src_pts, dst_pts = get_src_dst_pts(*init_edge, queryIdx_init, trainIdx_init, image_data)

    # Get pose and add cameras to graph
    _, _, _, R_out, t_out = get_pose(src_pts, dst_pts)
    stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
    pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
    inliers = filter_inliers(pts_3D_out, dst_pts, R_out, t_out)
    inliers = filter_inliers(pts_3D_out, src_pts, np.eye(3), np.zeros(3), inliers=inliers)

    G_covisibility.add_edges_from([init_edge], num_pts_3D=sum(inliers), stereo_camera=stereo_camera, processed=True)
    assert len(pts_3D_out) == len(track_objs) == len(inliers)
    for pt_3D, track_obj, is_inlier, src, dst in zip(pts_3D_out, track_objs, inliers, src_pts, dst_pts):
        track_obj.update_triangulated(*init_edge, pt_3D, is_inlier, src, dst)

    logger.info(f"Edge: {init_edge} has {sum(inliers)} inliers.")
    # plot_matches(image_data[init_edge[0]], image_data[init_edge[1]], queryIdx_init, trainIdx_init, inliers)

    init_tracks = [track_obj for track_obj in tracks if track_obj.is_valid_track(n=1)]
    for edges in get_next_edge(G_covisibility, init_edge=init_edge):
        reference_edge, new_edge = edges
        edge_loc, index = get_edge_relation(reference_edge, new_edge)
        logger.info(
            f"Running loop for edge: {new_edge} with reference: {reference_edge} with num_pts: {G_covisibility[reference_edge[0]][reference_edge[1]]['num_pts_3D']}, weight: {G_covisibility[new_edge[0]][new_edge[1]]['weight']}"
        )

        queryIdx_tracks, trainIdx_tracks, pts_3D_base, valid_pts_3D, track_objs = get_query_and_train(
            new_edge=new_edge,
            tracks=tracks,
            camera2trackIDs=camera2trackIDs,
            is_initial_edge=False,
            reference_edge=reference_edge,
        )
        if queryIdx_tracks.size == 0:
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Not enough query/train points for {new_edge}.")
            continue

        src_pts_tracks, dst_pts_tracks = get_src_dst_pts(*new_edge, queryIdx_tracks, trainIdx_tracks, image_data)
        R_out, t_out, _ = get_pnp_pose(
            pts_3D_base, dst_pts_tracks if edge_loc == "successor" else src_pts_tracks, np.zeros(5), mask=valid_pts_3D
        )
        if R_out is None:
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Could not retrieve valid pose for {new_edge}.")
            continue

        queryIdx_full, trainIdx_full = G_covisibility.edges[new_edge]["query_train"]
        src_pts, dst_pts = get_src_dst_pts(*new_edge, queryIdx_full, trainIdx_full, image_data)

        reference_stereo_camera: StereoCamera = G_covisibility.edges[reference_edge]["stereo_camera"]
        camera_poses = {
            "R_1": R_out if edge_loc == "successor" else reference_stereo_camera[index].R,
            "t_1": t_out if edge_loc == "successor" else reference_stereo_camera[index].t,
            "R_0": reference_stereo_camera[index].R if edge_loc == "successor" else R_out,
            "t_0": reference_stereo_camera[index].t if edge_loc == "successor" else t_out,
        }
        stereo_camera = StereoCamera(**camera_poses)
        pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D_out, dst_pts, camera_poses["R_1"], camera_poses["t_1"])
        inliers = filter_inliers(pts_3D_out, src_pts, camera_poses["R_0"], camera_poses["t_0"], inliers=inliers)

        G_covisibility.add_edges_from([new_edge], num_pts_3D=sum(inliers), stereo_camera=stereo_camera)
        common_track_IDs = get_common_track_IDs(new_edge, camera2trackIDs)

        for _q, _pt_3D, _inlier, src, dst in zip(queryIdx_full, pts_3D_out, inliers, src_pts, dst_pts):
            tracks_w_cam_and_query = node2trackID[(new_edge[0], _q)]
            for _track_id in tracks_w_cam_and_query:
                track_obj = tracks[_track_id]
                if new_edge[1] in track_obj.cameras:
                    track_obj.update_triangulated(*new_edge, _pt_3D, _inlier, src, dst)

        # queryIdx2point3d = dict(zip(queryIdx_full, pts_3D_out))
        # queryIdx2inliers = dict(zip(queryIdx_full, inliers))
        # _cam = new_edge[0]
        # for track_id in common_track_IDs:
        #     track_obj = tracks[track_id]
        #     _queryIdx = track_obj.cam2featureIdx[_cam]

        #     _pt_3D = queryIdx2point3d.get(_queryIdx, None)
        #     _inlier = queryIdx2inliers.get(_queryIdx, None)
        #     if _pt_3D is None:
        #         continue
        #     track_obj.update_triangulated(*new_edge, _pt_3D, _inlier)
        # print(f"Updated: {_num} tracks out of {len(common_track_IDs)} common tracks and {len(queryIdx_full)} queries.")

        logger.info(f"Edge: {new_edge} has {sum(inliers)} inliers.")
        # plot_matches(image_data[new_edge[0]], image_data[new_edge[1]], queryIdx_full, trainIdx_full, inliers)

    valid_tracks = [track_obj for track_obj in tracks if track_obj.is_valid_track(n=2)]
    print("NUMBER OF VALID TRACKS:", len(valid_tracks))
    setup_graph(
        G_covisibility,
        init_tracks,
        valid_tracks=valid_tracks,
    )

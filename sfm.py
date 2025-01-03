import argparse
import time

import networkx as nx
import numpy as np
import tqdm

from covisibility import (
    add_to_covisibility_graph,
    add_to_tracks_graph,
    get_common_track_IDs,
    get_edge_with_largest_weight,
    get_next_edge,
    get_query_and_train,
    obtrain_tracks,
)
from extract_features import get_image_loader, get_image_matcher_loader
from utils import (
    ImageData,
    StereoCamera,
    compute_essential_matrix,
    get_edge_relation,
    get_pnp_pose,
    get_pose,
    get_src_dst_pts,
    logger,
)
from visualization import plot_matches

IMG_DIR = "/home/aiuser/gonzag124/projects/SFM/buddha_images"


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    image_data: list[ImageData] = []
    G_tracks = nx.DiGraph()
    G_covisibility = nx.DiGraph()

    start = time.time()
    logger.info("Extracting local features.")

    img_loader = get_image_loader(IMG_DIR)
    for idx, (img, clahe, kpt, descriptors) in enumerate(tqdm.tqdm(img_loader, desc="Extracting features")):
        image_data.append(
            ImageData(image=img, rgb=img, filename=f"image_{idx}", keypoints=kpt, features=descriptors, clahe_img=clahe)
        )
    logger.info(f"Completed feature extraction for {len(image_data)} images.")

    pair_loader = get_image_matcher_loader(image_data)
    for idx, [(i, j, matches)] in enumerate(tqdm.tqdm(pair_loader, desc="Matching features")):
        G_tracks = add_to_tracks_graph(G_tracks, i, j, matches)
        G_covisibility = add_to_covisibility_graph(G_covisibility, i, j, matches)

    end = time.time()
    logger.info(f"Feature extraction and matching time taken: {end - start} seconds.")

    start = time.time()
    tracks, camera2trackIDs = obtrain_tracks(G_tracks)
    end = time.time()
    logger.info(f"Track extraction time taken: {end - start} seconds.")

    # Get initial edge
    start = time.time()
    init_edge = (0, 1)  # get_edge_with_largest_weight(G_covisibility)
    queryIdx, trainIdx, _, _, track_objs = get_query_and_train(
        new_edge=init_edge, tracks=tracks, camera2trackIDs=camera2trackIDs, is_initial_edge=True
    )

    # Get corresponding source and destination points
    src_pts, dst_pts = get_src_dst_pts(*init_edge, queryIdx, trainIdx, image_data)

    # Get pose and add cameras to graph
    _, _, inliers, R_out, t_out = get_pose(src_pts, dst_pts)
    stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
    pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
    inliers = inliers & (pts_3D_out[:, -1] > 0)

    G_covisibility.add_edges_from(
        [init_edge], num_pts_3D=sum(inliers), stereo_camera=stereo_camera, inliers=inliers, processed=True
    )
    assert len(pts_3D_out) == len(track_objs) == len(inliers)
    for pt_3D, track_obj, is_inlier in zip(pts_3D_out, track_objs, inliers):
        track_obj.update_triangulated(*init_edge, pt_3D, is_inlier)

    end = time.time()
    logger.info(f"Initialization of edge time taken: {end - start} seconds.")
    # plot_matches(
    #     image_data[init_edge[0]], image_data[init_edge[1]], queryIdx, trainIdx, inliers, "-".join(map(str, init_edge))
    # )

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
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1, processed=True)
            logger.info(f"Not enough query/train points for {new_edge}.")
            continue

        src_pts_tracks, dst_pts_tracks = get_src_dst_pts(*new_edge, queryIdx_tracks, trainIdx_tracks, image_data)
        R_out, t_out, __ = get_pnp_pose(
            pts_3D_base, dst_pts_tracks if edge_loc == "successor" else src_pts_tracks, np.zeros(5), mask=valid_pts_3D
        )
        if R_out is None:
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1, processed=True)
            logger.info(f"Could not retrieve valid pose for {new_edge}.")
            continue

        queryIdx, trainIdx = G_covisibility.edges[new_edge]["query_train"]
        src_pts, dst_pts = get_src_dst_pts(*new_edge, queryIdx, trainIdx, image_data)
        _, _, inliers = compute_essential_matrix(src_pts, dst_pts)

        reference_stereo_camera: StereoCamera = G_covisibility.edges[reference_edge]["stereo_camera"]
        camera_poses = {
            "R_1": R_out if edge_loc == "successor" else reference_stereo_camera[index].R,
            "t_1": t_out if edge_loc == "successor" else reference_stereo_camera[index].t,
            "R_0": reference_stereo_camera[index].R if edge_loc == "successor" else R_out,
            "t_0": reference_stereo_camera[index].t if edge_loc == "successor" else t_out,
        }
        stereo_camera = StereoCamera(**camera_poses)
        pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = inliers & (pts_3D_out[:, -1] > 0)

        G_covisibility.add_edges_from(
            [new_edge], num_pts_3D=sum(inliers), stereo_camera=stereo_camera, inliers=inliers, processed=True
        )

        # TODO: map featureIdx to track_id for each camera -> {cam: {featureIdx: track_id}} and do some assertions
        common_track_IDs = get_common_track_IDs(new_edge, camera2trackIDs)
        queryIdx2point3d = dict(zip(queryIdx, pts_3D_out))
        queryIdx2inliers = dict(zip(queryIdx, inliers))
        _cam = new_edge[0]
        _num = 0
        for track_id in tqdm.tqdm(common_track_IDs):
            track_obj = tracks[track_id]
            _queryIdx = track_obj.cam2featureIdx[_cam]

            _pt_3D = queryIdx2point3d.get(_queryIdx, None)
            _inlier = queryIdx2inliers.get(_queryIdx, None)

            if _pt_3D is None or _inlier is None:
                _num += 1
                # TODO: This is concerning
                continue
            track_obj.update_triangulated(*new_edge, _pt_3D, _inlier)

        print(f"Number of tracks not updated: {_num} out of {len(common_track_IDs)}")
        # plot_matches(
        #     image_data[new_edge[0]], image_data[new_edge[1]], queryIdx, trainIdx, inliers, "-".join(map(str, new_edge))
        # )
    valid_tracks = [track_obj for track_obj in tracks if track_obj.is_valid_track()]
    print("NUMBER OF VALID TRACKS: ", len(valid_tracks))

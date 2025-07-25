import argparse
import os
import sys
from pathlib import Path

import natsort
import networkx as nx
import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from sfm.config import CONFIG
from sfm.covisibility import (
    Track,
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
from sfm.sparse_reconstruction import optimize
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
            ImageData(
                image=img,
                rgb=img,
                filename=f"image_{idx}",
                keypoints=kpt,
                features=descriptors,
                clahe_img=clahe,
            )
        )

    pair_loader = get_image_matcher_loader(image_data)
    for idx, [(camera_0, camera_1, matches)] in enumerate(tqdm.tqdm(pair_loader, desc="Matching features")):
        G_tracks = add_to_tracks_graph(G_tracks, camera_0, camera_1, matches)
        G_covisibility = add_to_covisibility_graph(G_covisibility, camera_0, camera_1, matches)

    return G_tracks, G_covisibility, image_data


def update_tracks(
    tracks: list[Track],
    node2trackID: dict[tuple[int, int], list[int]],
    new_edge: tuple[int, int],
    queryIdx: np.ndarray,
    pts_3D: np.ndarray,
    inliers: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
):
    assert len(queryIdx) == len(pts_3D) == len(inliers) == len(src_pts) == len(dst_pts)
    for query, pt_3D, inlier, src, dst in zip(queryIdx, pts_3D, inliers, src_pts, dst_pts):
        trackIDs_from_cam_and_query = node2trackID[(new_edge[0], query)]
        for track_id in trackIDs_from_cam_and_query:
            track_obj = tracks[track_id]
            if new_edge[1] in track_obj.cameras:
                track_obj.update_triangulated(*new_edge, pt_3D, inlier, src, dst)

    # common_track_IDs = get_common_track_IDs(new_edge, camera2trackIDs)
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


def run_reconstruction(
    image_data: list[ImageData],
    G_covisibility: nx.DiGraph,
    init_edge: tuple[int, int],
    tracks: list[Track],
    camera2trackIDs: dict[int, list[int]],
    node2trackID: dict[tuple[int, int], list[int]],
    plot_images: bool = False,
):

    # Get corresponding source and destination points
    query_train_tracks = get_query_and_train(init_edge, tracks, camera2trackIDs)
    src_pts, dst_pts = get_src_dst_pts(
        *init_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks, image_data
    )

    # Get pose of initial 2 cameras and triangulate to get 3D points
    _, _, _, R_out, t_out = get_pose(src_pts, dst_pts)
    stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
    pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
    inliers = filter_inliers(pts_3D_out, dst_pts, R_out, t_out)
    inliers = filter_inliers(pts_3D_out, src_pts, np.eye(3), np.zeros(3), inliers=inliers)

    logger.info(f"Edge: {init_edge} has {sum(inliers)} inliers.")
    if plot_images:
        plot_matches(
            image_data[init_edge[0]],
            image_data[init_edge[1]],
            query_train_tracks.queryIdx_tracks,
            query_train_tracks.trainIdx_tracks,
            inliers,
        )

    # Update G_covisibility and relevant tracks
    G_covisibility.add_edges_from(
        [init_edge],
        num_pts_3D=sum(inliers),
        stereo_camera=stereo_camera,
        processed=True,
    )
    update_tracks(
        tracks=tracks,
        node2trackID=node2trackID,
        new_edge=init_edge,
        queryIdx=query_train_tracks.queryIdx_tracks,
        pts_3D=pts_3D_out,
        inliers=inliers,
        src_pts=src_pts,
        dst_pts=dst_pts,
    )

    for ref_num_pts, ref_weight, reference_edge, new_edge in get_next_edge(G_covisibility, init_edge=init_edge):
        edge_loc, index = get_edge_relation(reference_edge, new_edge)
        logger.info(
            f"Evaluating edge: {new_edge} with reference: {reference_edge} with num_pts: {-ref_num_pts} and weight: {-ref_weight}"
        )

        # Get corresponding source and destination points that span all cameras in reference_edge and new_edge
        query_train_tracks = get_query_and_train(
            new_edge=new_edge,
            tracks=tracks,
            camera2trackIDs=camera2trackIDs,
            is_initial_edge=False,
            reference_edge=reference_edge,
        )
        if query_train_tracks.queryIdx_tracks.size == 0:
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Not enough query/train points for {new_edge}.")
            continue

        # Get pose of the new camera using the reference triangulated points for PnP
        src_pts_tracks, dst_pts_tracks = get_src_dst_pts(
            *new_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks, image_data
        )

        R_out, t_out, _ = get_pnp_pose(
            query_train_tracks.pts_3D_ref,
            dst_pts_tracks if edge_loc == "successor" else src_pts_tracks,
            np.zeros(5),
            mask=query_train_tracks.valid_pts_3D,
        )
        if R_out is None:
            G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Could not retrieve valid pose for {new_edge}.")
            continue

        # Get corresponding source and destination points between cameras in new_edge
        queryIdx_full, trainIdx_full = G_covisibility.edges[new_edge]["query_train"]
        src_pts, dst_pts = get_src_dst_pts(*new_edge, queryIdx_full, trainIdx_full, image_data)

        # Triangulate to get 3D points
        reference_stereo_camera: StereoCamera = G_covisibility.edges[reference_edge]["stereo_camera"]
        camera_poses = {
            "R_1": (R_out if edge_loc == "successor" else reference_stereo_camera[index].R),
            "t_1": (t_out if edge_loc == "successor" else reference_stereo_camera[index].t),
            "R_0": (reference_stereo_camera[index].R if edge_loc == "successor" else R_out),
            "t_0": (reference_stereo_camera[index].t if edge_loc == "successor" else t_out),
        }
        stereo_camera = StereoCamera(**camera_poses)
        pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D_out, dst_pts, camera_poses["R_1"], camera_poses["t_1"])
        inliers = filter_inliers(
            pts_3D_out,
            src_pts,
            camera_poses["R_0"],
            camera_poses["t_0"],
            inliers=inliers,
        )
        logger.info(f"Edge: {new_edge} has {sum(inliers)} inliers.")

        # Update G_covisibility and relevant tracks
        G_covisibility.add_edges_from([new_edge], num_pts_3D=sum(inliers), stereo_camera=stereo_camera)
        update_tracks(
            tracks=tracks,
            node2trackID=node2trackID,
            new_edge=new_edge,
            queryIdx=queryIdx_full,
            pts_3D=pts_3D_out,
            inliers=inliers,
            src_pts=src_pts,
            dst_pts=dst_pts,
        )

        if plot_images:
            plot_matches(
                image_data[new_edge[0]],
                image_data[new_edge[1]],
                queryIdx_full,
                trainIdx_full,
                inliers,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nfeatures",
        type=int,
        default=CONFIG.nfeatures,
        help="Number of keypoints to detect using SIFT (default: %(default)s).",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        default=CONFIG.nlayers,
        help="Number of octave layers for SIFT keypoint detection (default: %(default)s).",
    )
    parser.add_argument(
        "--contrastThreshold",
        type=float,
        default=CONFIG.contrastThreshold,
        help="Contrast threshold for SIFT (default: %(default)s).",
    )
    parser.add_argument(
        "--edgeThreshold",
        type=float,
        default=CONFIG.edgeThreshold,
        help="Edge threshold for SIFT (default: %(default)s).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=CONFIG.sigma,
        help="Sigma for Gaussian blur in SIFT (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIG.threshold,
        help="Threshold for feature matching; controls ratio test filtering (default: %(default)s).",
    )
    parser.add_argument(
        "--ransacReprojThreshold",
        type=float,
        default=CONFIG.ransacReprojThreshold,
        help="Reprojection error threshold for RANSAC in Essential matrix computation (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIG.confidence,
        help="Confidence level for RANSAC in Essential matrix computation (default: %(default)s).",
    )
    parser.add_argument(
        "--RGB",
        type=bool,
        default=CONFIG.RGB,
        help="Whether to use RGB images (True) or grayscale images (False) (default: %(default)s).",
    )
    parser.add_argument(
        "--clahe",
        type=bool,
        default=CONFIG.clahe,
        help="Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) (default: %(default)s).",
    )
    parser.add_argument(
        "--NMS",
        type=bool,
        default=CONFIG.NMS,
        help="Whether to apply Non-Maximum Suppression to keypoints (default: %(default)s).",
    )
    parser.add_argument(
        "--NMS_dist",
        type=int,
        default=CONFIG.NMS_dist,
        help="Distance threshold for Non-Maximum Suppression (default: %(default)s).",
    )
    parser.add_argument(
        "--clipLimit",
        type=float,
        default=CONFIG.clipLimit,
        help="Clip limit for CLAHE to prevent over-amplification of noise (default: %(default)s).",
    )
    parser.add_argument(
        "--tileGridSize",
        type=tuple,
        default=CONFIG.tileGridSize,
        help="Tile grid size for CLAHE, helps control spatial histogram equalization (default: %(default)s).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=CONFIG.method,
        help="Method to use for track extraction (default: %(default)s).",
    )

    args = parser.parse_args()
    CONFIG.update(vars(args))

    # Preprocessing
    G_tracks, G_covisibility, image_data = preprocess()
    tracks, camera2trackIDs, node2trackID = obtrain_tracks(G_tracks, method=CONFIG.method)
    # TODO: Not sure I like node2trackID

    # Get initial edge
    init_edge = (7, 8)  # get_edge_with_largest_weight(G_covisibility)

    # Run reconstruction
    run_reconstruction(
        image_data=image_data,
        G_covisibility=G_covisibility,
        init_edge=init_edge,
        tracks=tracks,
        camera2trackIDs=camera2trackIDs,
        node2trackID=node2trackID,
    )

    # Optimize
    valid_tracks = [track_obj for track_obj in tracks if track_obj.is_valid_track(n=2)]
    valid_tracks = natsort.natsorted(valid_tracks, key=lambda x: x.valid_pts)
    logger.info(f"Number of valid tracks: {len(valid_tracks)}")
    optimize(
        G_covisibility,
        valid_tracks=valid_tracks,
    )

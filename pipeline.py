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
from sfm.covisibility import CovisibilityGraph, QueryTrainTracks
from sfm.extract_features import get_image_loader
from sfm.sparse_reconstruction import optimize
from sfm.utils import (
    ImageData,
    StereoCamera,
    filter_inliers,
    get_edge_relation,
    get_pnp_pose,
    get_pose,
    logger,
)
from sfm.visualization import plot_matches


class SFMPipeline(CovisibilityGraph):
    def __init__(self, image_dir: str, plot_images: bool = False, track_extraction_method: str = "dijkstra"):

        self.image_dir = image_dir
        self.plot_images = plot_images
        self.track_extraction_method = track_extraction_method

        self.image_data: list[ImageData] = []
        self.__preprocess()

    def __preprocess(self) -> tuple[nx.DiGraph, nx.DiGraph, list[ImageData]]:

        img_loader = get_image_loader(self.image_dir)
        for idx, (img, clahe, kpt, descriptors) in enumerate(tqdm.tqdm(img_loader, desc="Extracting features")):
            self.image_data.append(
                ImageData(
                    image=img,
                    rgb=img,
                    filename=f"image_{idx}",
                    keypoints=kpt,
                    features=descriptors,
                    clahe_img=clahe,
                )
            )
        super().__init__(self.image_data, self.track_extraction_method)

    def __get_src_dst_pts(
        self,
        camera_0: int,
        camera_1: int,
        queryIdx: np.ndarray,
        trainIdx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_0 = self.image_data[camera_0]
        image_1 = self.image_data[camera_1]

        src_pts = image_0.keypoints_np[queryIdx]
        dst_pts = image_1.keypoints_np[trainIdx]
        assert len(src_pts) == len(dst_pts), "Length of queryIdx and trainIdx need to match!"
        return src_pts, dst_pts

    def __get_cam_pose_from_reference(
        self,
        new_edge: tuple[int, int],
        is_initial_edge: bool,
        query_train_tracks: QueryTrainTracks,
        edge_loc: str | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Get pose of the new camera using the reference triangulated points for PnP.

        Parameters
        ----------
        new_edge : tuple[int, int]
            _description_
        is_initial_edge : bool
            _description_
        query_train_tracks : QueryTrainTracks
            _description_
        edge_loc : str
            _description_

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Rotation and translation of new camera
        """
        src_pts_tracks, dst_pts_tracks = self.__get_src_dst_pts(
            *new_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks
        )

        if is_initial_edge:
            # Get pose using Essential Matrix
            _, _, _, R_out, t_out = get_pose(src_pts_tracks, dst_pts_tracks)
        else:
            # Get pose using PnP
            R_out, t_out, _ = get_pnp_pose(
                query_train_tracks.pts_3D_ref,
                dst_pts_tracks if edge_loc == "successor" else src_pts_tracks,
                np.zeros(5),
                mask=query_train_tracks.valid_pts_3D,
            )
        if R_out is None:
            self.G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Could not retrieve valid pose for {new_edge}.")
            return None, None

        return R_out, t_out

    def __triangulate_from_reference(
        self,
        new_edge: tuple[int, int],
        is_initial_edge: bool,
        R_out: np.ndarray,
        t_out: np.ndarray,
        reference_edge: tuple[int, int] | None,
        edge_loc: str | None,
        index: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StereoCamera]:
        """
        Obtains matched source and destination points between cameras in new_edge. Uses a reference edge
        to get an established StereoCamera that is used to triangulate the source and destination points
        as stereo cameras.

        Parameters
        ----------
        new_edge : tuple[int, int]
            _description_
        is_initial_edge : bool
            _description_
        R_out : np.ndarray
            _description_
        t_out : np.ndarray
            _description_
        reference_edge : tuple[int, int] | None
            _description_
        edge_loc : str | None
            _description_
        index : int | None
            _description_

        Returns
        -------
        tuple[np.ndarray, StereoCamera]
            Inliers and new stereo camera.
        """
        # Get corresponding source and destination points between cameras in new_edge
        queryIdx_full, trainIdx_full = self.G_covisibility.edges[new_edge]["query_train"]
        src_pts, dst_pts = self.__get_src_dst_pts(*new_edge, queryIdx_full, trainIdx_full)

        # Triangulate to get 3D points
        if is_initial_edge:
            camera_poses = {"R_1": R_out, "t_1": t_out, "R_0": np.eye(3), "t_0": np.zeros(3)}
        else:
            reference_stereo_camera: StereoCamera = self.G_covisibility.edges[reference_edge]["stereo_camera"]
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
        return queryIdx_full, trainIdx_full, inliers, pts_3D_out, src_pts, dst_pts, stereo_camera

    def run(self, init_edge: tuple[int, int]):

        query_train_tracks = super().get_camera_span_from_edge(init_edge, True, None)
        src_pts, dst_pts = self.__get_src_dst_pts(
            *init_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks
        )

        # Get pose of initial 2 cameras and triangulate to get 3D points
        _, _, _, R_out, t_out = get_pose(src_pts, dst_pts)
        stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
        pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D_out, dst_pts, R_out, t_out)
        inliers = filter_inliers(pts_3D_out, src_pts, np.eye(3), np.zeros(3), inliers=inliers)

        logger.info(f"Edge: {init_edge} has {sum(inliers)} inliers.")
        if self.plot_images:
            plot_matches(
                self.image_data[init_edge[0]],
                self.image_data[init_edge[1]],
                query_train_tracks.queryIdx_tracks,
                query_train_tracks.trainIdx_tracks,
                inliers,
            )

        # Update tracks with init_edge
        attrs = dict(num_pts_3D=sum(inliers), stereo_camera=stereo_camera, processed=True)
        self.G_covisibility.add_edges_from([init_edge], **attrs)
        self.tracks.update_tracks(init_edge, query_train_tracks.queryIdx_tracks, pts_3D_out, inliers, src_pts, dst_pts)

        for ref_num_pts, ref_weight, reference_edge, new_edge in super().get_next_edge(init_edge):
            logger.info(f"Evaluating edge: {new_edge} with reference: {reference_edge}")
            edge_loc, index = get_edge_relation(reference_edge, new_edge)
            logger.info(f"Details - num_pts: {-ref_num_pts} and weight: {-ref_weight}")

            query_train_tracks = super().get_camera_span_from_edge(new_edge, False, reference_edge)
            if query_train_tracks is None:
                continue

            R_out, t_out = self.__get_cam_pose_from_reference(new_edge, False, query_train_tracks, edge_loc)
            if R_out is None:
                continue

            queryIdx_full, trainIdx_full, inliers, pts_3D_out, src_pts, dst_pts, stereo_camera = (
                self.__triangulate_from_reference(new_edge, False, R_out, t_out, reference_edge, edge_loc, index)
            )

            # Update tracks with new_edge
            attrs = dict(num_pts_3D=sum(inliers), stereo_camera=stereo_camera)
            self.G_covisibility.add_edges_from([new_edge], **attrs)
            self.tracks.update_tracks(new_edge, queryIdx_full, pts_3D_out, inliers, src_pts, dst_pts)

            if self.plot_images:
                plot_matches(
                    self.image_data[new_edge[0]],
                    self.image_data[new_edge[1]],
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

    __cwd__ = os.path.dirname(__file__)

    sfm = SFMPipeline(os.path.join(__cwd__, "images"))
    sfm.run(init_edge=(7, 8))

    # Optimize
    valid_tracks = [track_obj for track_obj in sfm.tracks if track_obj.is_valid_track(n=2)]
    valid_tracks = natsort.natsorted(valid_tracks, key=lambda x: x.valid_pts)
    logger.info(f"Number of valid tracks: {len(valid_tracks)}")
    optimize(sfm.G_covisibility, valid_tracks=valid_tracks)

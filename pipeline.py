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
    QueryTrainTracks,
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
    get_pnp_pose,
    get_pose,
    get_src_dst_pts,
    logger,
)
from sfm.visualization import plot_matches


class SFMPipeline:
    def __init__(self, image_dir: str, plot_images: bool = False, track_extraction_method: str = "dijkstra"):

        self.image_dir = image_dir
        self.plot_images = plot_images
        self.track_extraction_method = track_extraction_method

        self.image_data: list[ImageData] = []
        self.G_tracks = nx.DiGraph()
        self.G_covisibility = nx.DiGraph()

        self.__preprocess()
        self.__obtain_tracks()

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

        pair_loader = get_image_matcher_loader(self.image_data)
        for idx, [(camera_0, camera_1, matches)] in enumerate(tqdm.tqdm(pair_loader, desc="Matching features")):
            self.G_tracks = add_to_tracks_graph(self.G_tracks, camera_0, camera_1, matches)
            self.G_covisibility = add_to_covisibility_graph(self.G_covisibility, camera_0, camera_1, matches)

    def __obtain_tracks(self):
        """
        Obtain the corresponding tracks of feature points. Results are similar
        to those of: https://imagine.enpc.fr/~moulonp/publis/poster_CVMP12.pdf.

        Parameters
        ----------
        G : nx.DiGraph
            Graph where each node is (camera_id, feature_id).

        Returns
        -------
        tuple[list[Track], dict]
            List of track objects, where each track object contains a sorted list of
            tuples (camera_id, feature_id) by camera_id.
            Mapping from camera_id to all track IDs (via indices) that contain camera_id.
            Mapping of (camera_id, feature_id) to the track ID.
        """
        self.tracks, self.camera2trackIDs, self.node2trackID = obtrain_tracks(
            G=self.G_tracks, method=self.track_extraction_method
        )

    def __get_edge_relation(self, new_edge: tuple[int, int], reference_edge: tuple[int, int]) -> tuple[str, int]:
        """
        Get the relation of the new edge with respect to the reference edge. Relations can go as follows:

            - New camera (new_edge_1) is successor from reference's index 1
            [ref_edge_0, ref_edge_1] ----> [new_edge_0, new_edge_1]

            - New camera (new_edge_0) is predecessor from reference's index 1
            [ref_edge_0, ref_edge_1]
                            |
                            ↓
            [new_edge_0, new_edge_1]

            - New camera (new_edge_1) is successor from reference's index 0
            [ref_edge_0, ref_edge_1]
                |
                ↓
            [new_edge_0, new_edge_1]

            - New camera (new_edge_0) is predecessor from reference's index 0
            [ref_edge_0, ref_edge_1]
                |
                ------------
                            ↓
            [new_edge_0, new_edge_1]

        Parameters
        ----------
        new_edge : tuple[int, int]
            _description_
        reference_edge : tuple[int, int]
            _description_

        Returns
        -------
        tuple[str, int]
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """

        if not isinstance(reference_edge, tuple):
            raise ValueError(f"Expected reference_edge to be a tuple, got {type(reference_edge)} instead.")
        if not isinstance(new_edge, tuple):
            raise ValueError(f"Expected new_edge to be a tuple, got {type(new_edge)} instead.")

        shared_cameras = set(reference_edge).intersection(new_edge)
        assert len(shared_cameras) == 1, f"Expected one shared camera between {reference_edge} and {new_edge}."
        shared_camera = shared_cameras.pop()

        index = reference_edge.index(shared_camera)
        edge_loc = "successor" if shared_camera == new_edge[0] else "predecessor"

        return edge_loc, index

    def __get_camera_span_from_edge(
        self,
        new_edge: tuple[int, int],
        is_initial_edge: bool,
        reference_edge: tuple[int, int] = None,
    ) -> QueryTrainTracks | None:
        """
        Get corresponding source and destination points that span all cameras in reference_edge and new_edge.

        Parameters
        ----------
        new_edge : tuple[int, int]
            _description_
        is_initial_edge : bool
            _description_
        reference_edge : tuple[int, int], optional
            _description_, by default None

        Returns
        -------
        QueryTrainTracks | None
            _description_
        """
        query_train_tracks = get_query_and_train(
            new_edge=new_edge,
            tracks=self.tracks,
            camera2trackIDs=self.camera2trackIDs,
            is_initial_edge=is_initial_edge,
            reference_edge=reference_edge,
        )
        if query_train_tracks.queryIdx_tracks.size == 0:
            self.G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Not enough query/train points for {new_edge}.")
            return None

        return query_train_tracks

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
        src_pts_tracks, dst_pts_tracks = get_src_dst_pts(
            *new_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks, self.image_data
        )
        breakpoint()

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
        src_pts, dst_pts = get_src_dst_pts(*new_edge, queryIdx_full, trainIdx_full, self.image_data)

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

        breakpoint()
        logger.info(f"Edge: {new_edge} has {sum(inliers)} inliers.")  # FIXME: THIS IS GIVING LOWER NUMBER
        return queryIdx_full, trainIdx_full, inliers, pts_3D_out, src_pts, dst_pts, stereo_camera

    def __update_tracks(
        self,
        new_edge: tuple[int, int],
        stereo_camera: StereoCamera,
        queryIdx_full: np.ndarray,
        pts_3D_out: np.ndarray,
        inliers: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        processed: bool = False,
    ):

        attrs = dict(num_pts_3D=sum(inliers), stereo_camera=stereo_camera)
        if processed:
            attrs["processed"] = processed
        self.G_covisibility.add_edges_from([new_edge], **attrs)

        assert len(queryIdx_full) == len(pts_3D_out) == len(inliers) == len(src_pts) == len(dst_pts)
        for query, pt_3D, inlier, src, dst in zip(queryIdx_full, pts_3D_out, inliers, src_pts, dst_pts):
            trackIDs_from_cam_and_query = self.node2trackID[(new_edge[0], query)]
            for track_id in trackIDs_from_cam_and_query:
                track_obj = self.tracks[track_id]
                if new_edge[1] in track_obj.cameras:
                    track_obj.update_triangulated(*new_edge, pt_3D, inlier, src, dst)

    def run(self, init_edge: tuple[int, int]):

        query_train_tracks = self.__get_camera_span_from_edge(init_edge, True, None)
        src_pts, dst_pts = get_src_dst_pts(
            *init_edge, query_train_tracks.queryIdx_tracks, query_train_tracks.trainIdx_tracks, self.image_data
        )
        breakpoint()

        # Get pose of initial 2 cameras and triangulate to get 3D points
        _, _, _, R_out, t_out = get_pose(src_pts, dst_pts)
        stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
        pts_3D_out = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D_out, dst_pts, R_out, t_out)
        inliers = filter_inliers(pts_3D_out, src_pts, np.eye(3), np.zeros(3), inliers=inliers)

        breakpoint()
        logger.info(f"Edge: {init_edge} has {sum(inliers)} inliers.")
        if self.plot_images:
            plot_matches(
                self.image_data[init_edge[0]],
                self.image_data[init_edge[1]],
                query_train_tracks.queryIdx_tracks,
                query_train_tracks.trainIdx_tracks,
                inliers,
            )
        self.__update_tracks(
            init_edge,
            stereo_camera,
            query_train_tracks.queryIdx_tracks,
            pts_3D_out,
            inliers,
            src_pts,
            dst_pts,
            processed=True,
        )

        for ref_num_pts, ref_weight, reference_edge, new_edge in get_next_edge(self.G_covisibility, init_edge):
            logger.info(f"Evaluating edge: {new_edge} with reference: {reference_edge}")
            edge_loc, index = self.__get_edge_relation(reference_edge, new_edge)
            logger.info(f"Details - num_pts: {-ref_num_pts} and weight: {-ref_weight}")

            query_train_tracks = self.__get_camera_span_from_edge(new_edge, False, reference_edge)
            if query_train_tracks is None:
                continue

            R_out, t_out = self.__get_cam_pose_from_reference(new_edge, False, query_train_tracks, edge_loc)
            if R_out is None:
                continue

            queryIdx_full, trainIdx_full, inliers, pts_3D_out, src_pts, dst_pts, stereo_camera = (
                self.__triangulate_from_reference(new_edge, False, R_out, t_out, reference_edge, edge_loc, index)
            )
            self.__update_tracks(new_edge, stereo_camera, queryIdx_full, pts_3D_out, inliers, src_pts, dst_pts)

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

    sfm = SFMPipeline("/run/media/gman/Elements/personal/DL-CV-ML Projects/SFM/images/")
    sfm.run(init_edge=(7, 8))

    # Optimize
    valid_tracks = [track_obj for track_obj in sfm.tracks if track_obj.is_valid_track(n=2)]
    valid_tracks = natsort.natsorted(valid_tracks, key=lambda x: x.valid_pts)
    logger.info(f"Number of valid tracks: {len(valid_tracks)}")
    optimize(sfm.G_covisibility, valid_tracks=valid_tracks)

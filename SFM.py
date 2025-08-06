import argparse
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from sfm.config import CONFIG
from sfm.covisibility import CovisibilityGraph, QueryTrainTracks
from sfm.extract_features import get_image_loader
from sfm.sparse_reconstruction import SceneGraph
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

        self.scene_graph = SceneGraph()

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
        camera_pair: tuple[int, int],
        queryIdx: np.ndarray,
        trainIdx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given the camera pairing whose point features were matched, will retrieve the query and
        train points.

        Given camera pair as (cam_i, cam_j), src_pts will come from cam_i and dst_pts will come from cam_j.

        Parameters
        ----------
        camera_pair : tuple[int, int]
            Camera pairing.
        queryIdx : np.ndarray
            Query points of first camera, used to filter out the 2D keypoints.
        trainIdx : np.ndarray
            Train points of second camera, used to filter out the 2D keypoints.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Source and destination points of the camera pairing.
        """
        image_0 = self.image_data[camera_pair[0]]
        image_1 = self.image_data[camera_pair[1]]

        src_pts = image_0.keypoints_np[queryIdx]
        dst_pts = image_1.keypoints_np[trainIdx]
        assert len(src_pts) == len(dst_pts), "Length of queryIdx and trainIdx need to match!"
        return src_pts, dst_pts

    def __get_cam_pose_from_reference(
        self,
        new_edge: tuple[int, int],
        ref2new_tracks: QueryTrainTracks,
        edge_loc: str | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Get pose of the new camera using the reference triangulated points for PnP.

        Parameters
        ----------
        new_edge : tuple[int, int]
            Camera pair: (cam_i, cam_j)
        ref2new_tracks : QueryTrainTracks
            Dataclass with info on tracks that contain the set of cameras in (*new_edge, *reference_edge).
        edge_loc : str
            "successor" if unseen camera in new_edge succeds from their shared camera, else "predecessor".

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Rotation and translation of new camera
        """
        # Get src_pts and dst_pts from (cam_i, cam_j) respectively
        src_pts_tracks, dst_pts_tracks = self.__get_src_dst_pts(
            new_edge, ref2new_tracks.queryIdx_tracks, ref2new_tracks.trainIdx_tracks
        )

        # Get pose using PnP
        R_out, t_out, _ = get_pnp_pose(
            ref2new_tracks.pts_3D_ref,
            src_pts_tracks if edge_loc == "predecessor" else dst_pts_tracks,
            np.zeros(5),
            mask=ref2new_tracks.valid_pts_3D,
        )
        if R_out is None:
            self.G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            return None, None

        return R_out, t_out

    def __triangulate_from_reference(
        self,
        new_edge: tuple[int, int],
        queryIdx: np.ndarray,
        trainIdx: np.ndarray,
        R_out: np.ndarray,
        t_out: np.ndarray,
        reference_edge: tuple[int, int] | None,
        edge_loc: str | None,
        share_idx_from_ref: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StereoCamera]:
        """
        Obtains matched source and destination points between cameras in new_edge. Uses a reference edge
        to get an established StereoCamera that is used to triangulate the source and destination points
        as stereo cameras.

        Since the 3D points come from the tracked reference points, no need to iteratively update the
        pose as seen below. This is because a Track is meant to represent a single (global 3D) point
        that propagates from an initial image:

            T_ref = get_T(R_ref, t_ref)
            T_new = get_T(R_out, t_out)
            T_new = T_new @ T_ref # < Not needed!

        Parameters
        ----------
        new_edge : tuple[int, int]
            New camera pair: (cam_i, cam_j), to evaluate.
        queryIdx : np.ndarray
            Indices of the feature points for the image corresponding to camera=new_edge[0].
        trainIdx : np.ndarray
            Indices of the feature points for the image corresponding to camera=new_edge[1].
        R_out : np.ndarray
            Rotation of the newly unseen camera within new_edge.
        t_out : np.ndarray
            Translation of the newly unseen camera within new_edge.
        reference_edge : tuple[int, int] | None
            Reference edge for new_edge from covisibility graph.
        edge_loc : str | None
            "successor" if unseen camera in new_edge succeds from their shared camera, else "predecessor".
        share_idx_from_ref : int | None
            Index the shared camera between new_edge and reference_edge exists in reference_edge.

        Returns
        -------
        tuple[np.ndarray, StereoCamera]
            Inliers and new stereo camera.
        """
        # Get corresponding source and destination points between cameras in new_edge
        src_pts, dst_pts = self.__get_src_dst_pts(new_edge, queryIdx, trainIdx)

        reference_stereo_camera: StereoCamera = self.G_covisibility.edges[reference_edge]["stereo_camera"]
        R_ref = reference_stereo_camera[share_idx_from_ref].R
        t_ref = reference_stereo_camera[share_idx_from_ref].t

        camera_poses = {
            "R_1": (R_out if edge_loc == "successor" else R_ref),
            "t_1": (t_out if edge_loc == "successor" else t_ref),
            "R_0": (R_ref if edge_loc == "successor" else R_out),
            "t_0": (t_ref if edge_loc == "successor" else t_out),
        }

        stereo_camera = StereoCamera(**camera_poses)
        pts_3D = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D, dst_pts, camera_poses["R_1"], camera_poses["t_1"])
        inliers = filter_inliers(
            pts_3D,
            src_pts,
            camera_poses["R_0"],
            camera_poses["t_0"],
            inliers=inliers,
        )

        logger.info(f"Edge: {new_edge} has {sum(inliers)} inliers.")
        return inliers, pts_3D, src_pts, dst_pts, stereo_camera

    def run(self, init_edge: tuple[int, int], direct_update: bool = False, min_inliers: int = 10):

        init_tracks = super().get_camera_span_from_edge(init_edge, None)
        src_pts, dst_pts = self.__get_src_dst_pts(init_edge, init_tracks.queryIdx_tracks, init_tracks.trainIdx_tracks)

        # Get pose of initial 2 cameras and triangulate to get 3D points
        # Given this is the initialization step, R_out and t_out represent
        # the global pose of cam_j if init_edge = (cam_i, cam_j)
        _, _, _, R_out, t_out = get_pose(src_pts, dst_pts)
        stereo_camera = StereoCamera(R_1=R_out, t_1=t_out)
        pts_3D = stereo_camera.triangulate(src_pts, dst_pts)
        inliers = filter_inliers(pts_3D, dst_pts, R_out, t_out)
        inliers = filter_inliers(pts_3D, src_pts, np.eye(3), np.zeros(3), inliers=inliers)

        logger.info(f"Edge: {init_edge} has {sum(inliers)} inliers.")
        if self.plot_images:
            plot_matches(
                self.image_data[init_edge[0]],
                self.image_data[init_edge[1]],
                init_tracks.queryIdx_tracks,
                init_tracks.trainIdx_tracks,
                inliers,
            )

        # Update tracks with init_edge
        attrs = dict(num_pts_3D=sum(inliers), stereo_camera=stereo_camera, processed=True)
        self.G_covisibility.add_edges_from([init_edge], **attrs)
        self.tracks.update_tracks(init_tracks, init_edge, pts_3D, inliers, src_pts, dst_pts, None, None, False)

        for ref_num_pts, ref_weight, reference_edge, new_edge in super().get_next_edge(init_edge):
            logger.info(f"Evaluating edge: {new_edge} with reference: {reference_edge}")
            edge_loc, share_idx_from_ref = get_edge_relation(reference_edge, new_edge)
            logger.info(f"Details - num_pts: {-ref_num_pts} and weight: {-ref_weight}")

            ref2new_tracks = super().get_camera_span_from_edge(new_edge, reference_edge)
            tracks_current = super().get_camera_span_from_edge(new_edge, None)

            if ref2new_tracks is None:
                logger.info(f"No shared tracks between (ref) {reference_edge} and (new) {new_edge}")
                continue

            R_out, t_out = self.__get_cam_pose_from_reference(new_edge, ref2new_tracks, edge_loc)
            if R_out is None:
                logger.info(f"Could not retrieve valid pose for {new_edge}.")
                continue

            queryIdx, trainIdx, inliers_needed = tracks_current.queryIdx_tracks, tracks_current.trainIdx_tracks, 3
            if direct_update:
                (queryIdx, trainIdx), inliers_needed = self.G_covisibility.edges[new_edge]["query_train"], 2

            inliers, pts_3D, src_pts, dst_pts, stereo_camera = self.__triangulate_from_reference(
                new_edge, queryIdx, trainIdx, R_out, t_out, reference_edge, edge_loc, share_idx_from_ref
            )

            if sum(inliers) <= min_inliers:
                logger.info(f"Not enough inliers left over for {new_edge}.")
                self.G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
                continue

            # Update tracks with new_edge
            attrs = dict(num_pts_3D=sum(inliers), stereo_camera=stereo_camera)
            self.G_covisibility.add_edges_from([new_edge], **attrs)
            self.tracks.update_tracks(
                tracks_current, new_edge, pts_3D, inliers, src_pts, dst_pts, queryIdx, trainIdx, direct_update
            )
            if self.plot_images:
                plot_matches(self.image_data[new_edge[0]], self.image_data[new_edge[1]], queryIdx, trainIdx, inliers)

            self.scene_graph.update_graph(self.G_covisibility, reference_edge, new_edge, tracks_current, inliers_needed)
        self.scene_graph.solve()


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
        choices=["dijkstra", "dfs", "unique_dfs"],
        help="Method to use for track extraction (default: %(default)s).",
    )
    parser.add_argument(
        "--no_direct_update",
        action="store_false",
        help="Disable direct_update flag. This means any track that simply contains the camera ids are updated.",
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=10,
        help="Min inliers required after triangulation to accept results for updating (default: %(default)s).",
    )

    args = parser.parse_args()
    CONFIG.update(vars(args))

    __cwd__ = os.path.dirname(__file__)

    sfm = SFMPipeline(os.path.join(__cwd__, "images"), plot_images=False, track_extraction_method="dijkstra")
    # TODO: Work on running multiple initializations
    sfm.run(init_edge=(7, 8), direct_update=(direct_update := args.no_direct_update), min_inliers=args.min_inliers)

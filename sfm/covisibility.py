import heapq
import time
from collections import defaultdict
from dataclasses import dataclass

import natsort
import networkx as nx
import numpy as np
import tqdm
from easydict import EasyDict

from sfm.image_matching import DMatch, get_image_matcher_loader
from sfm.utils import ImageData, logger

__default_3D__ = EasyDict({"pt_3D": np.array([-1.0, -1.0, -1.0])})
__default_in__ = EasyDict({"is_inlier": False})


@dataclass
class QueryTrainTracks:
    queryIdx_tracks: np.ndarray
    trainIdx_tracks: np.ndarray
    pts_3D_ref: np.ndarray
    valid_pts_3D: np.ndarray
    track_objs: list["Track"]


class TrackEdge:

    def __init__(
        self,
        camera_0: int,
        camera_1: int,
        pt_3D: np.ndarray,
        is_inlier: bool,
        src: np.ndarray,
        dst: np.ndarray,
    ):
        self.camera_0 = camera_0
        self.camera_1 = camera_1

        self.pt_3D = pt_3D
        self.is_inlier = is_inlier
        self.src = src
        self.dst = dst

    def get_edge_info(self) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, bool]:
        return ((self.src, self.dst), self.pt_3D, self.is_inlier)


class Track:
    """
    In theory, represents a single (3D) point that propagates from an initial image to subsequent images.

    Track represented by [[cam_0, featureIdx], [cam_1, featureIdx], ..., [cam_N, featureIdx]]
    for a single point that propagates through various cameras with different featureIdx's that are matched.
    """

    def __init__(self, track_id: int, track: list[tuple[int, int]] | np.ndarray, valid_pts: int = 0):
        """
        Initialize Track instance.

        Parameters
        ----------
        track_id : int
            Unique ID for the track.
        track : list[tuple[int, int]] | np.ndarray
            Track represented by [[cam_0, featureIdx], [cam_1, featureIdx], ..., [cam_N, featureIdx]]
            for a single point that propagates through various cameras with different featureIdx's that are matched.
        valid_pts : int, optional
            Number of valid points, by default 0.
        """

        self.track_id = track_id
        self.track = track
        self.valid_pts = valid_pts
        self.__setup()

    def __setup(self):

        # Mapping from camera_id to feature index
        self.cam2featureIdx = dict(self.track)
        # Get the cameras in order seen in the track
        self.cameras = np.asarray(self.track)[:, 0]
        # Initialize mapping from edge ((cam_i, cam_j) ) to triangulated
        # boolean, 3D point, inlier status, and respective 2D points
        self._triangulated_edges: dict[tuple[int, int], TrackEdge] = defaultdict(
            lambda: TrackEdge(
                None, None, np.array([-1.0, -1.0, -1.0]), False, np.array([-1.0, -1.0]), np.array([-1.0, -1.0])
            )
        )
        # NOTE: Why do we need a default dict?
        # When updating the GTSAM graph, all the tracks that contain `new_cam_0` and `new_cam_1` are passed.
        # This includes tracks that may link like so: ... --> new_cam_0 --> new_cam_n --> new_cam_1 ... -->.
        # However, when updating the tracks, if `direct_update=True`, then only the tracks that have direct
        # links are updated, or tracks that only link like so: ... --> new_cam_0 --> new_cam_1 ... -->. This
        # means that some tracks will not contain (new_edge) when passed through the GTSAM update_graph(...)

    def update_triangulated(
        self,
        camera_0: int,
        camera_1: int,
        pt_3D: np.ndarray,
        is_inlier: bool,
        src: np.ndarray,
        dst: np.ndarray,
    ):
        _edge = (camera_0, camera_1)
        assert self.cam2featureIdx.get(camera_0, None) is not None, f"{camera_0} is not a valid camera in {self.track}."
        assert self.cam2featureIdx.get(camera_1, None) is not None, f"{camera_1} is not a valid camera in {self.track}."
        if is_inlier and not _edge in self._triangulated_edges:
            self.valid_pts += 1

        self._triangulated_edges[_edge] = TrackEdge(
            camera_0=camera_0,
            camera_1=camera_1,
            pt_3D=pt_3D,
            is_inlier=is_inlier,
            src=src,
            dst=dst,
        )

    def get_pt_3D(self, edge: tuple[int, int] = None) -> np.ndarray:
        return np.array([-1, -1, -1.0]) if edge is None else self._triangulated_edges.get(edge, __default_3D__).pt_3D

    def is_inlier(self, edge: tuple[int, int] = None) -> bool:
        return False if edge is None else self._triangulated_edges.get(edge, __default_in__).is_inlier

    def is_valid_track(self, n: int = 3) -> bool:
        return self.valid_pts >= n

    def get_edges(self) -> list[tuple[int, int]]:
        """
        Used to construct GTSAM optimization graph.

        Returns
        -------
        list[tuple[int, int]]
            Returns the edges [(cam_0, cam_1), ...]
        """
        return list(self._triangulated_edges.keys())

    def get_edge(self, edge: tuple[int, int]) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, bool]:
        """
        Retrieve stored data for the respective edge. Since this happens at post-processing, `edge`
        should be a valid key in self._triangulated_edges.

        "edge" in this context refers to the source-and-latest camera pairing. This function is
        primarily used to update/create GTSAM graph.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge consisting of camera_0 and camera_1.

        Returns
        -------
        tuple[tuple[np.ndarray, np.ndarray], np.ndarray, bool]
            Edge data such as source and destination points, 3D points, and inlier booleans
        """
        if edge[0] not in self.cameras:
            raise AssertionError(f"Camera {edge[0]} is not a valid camera in {self.cameras}")
        if edge[1] not in self.cameras:
            raise AssertionError(f"Camera {edge[1]} is not a valid camera in {self.cameras}")

        logger.debug(f"Cameras seen: {self.cameras}")
        logger.debug(f"Triangulated edges: {list(self._triangulated_edges.keys())}, on track: {self}")
        return self._triangulated_edges[edge].get_edge_info()

    def __len__(self) -> int:
        return len(self.track)

    def __repr__(self) -> str:
        return f"{self.track}"


class Tracks(list[Track]):

    def __init__(self):
        super().__init__()
        # Mapping from camera_id to all tracks (via indices) that contain camera_id
        self.camera2trackIDs: dict[int, list[int]] = defaultdict(list)
        # Mapping of (camera_id, feature_id) to the track ID
        self.node2trackIDs: dict[tuple[int, int], list[int]] = defaultdict(list)  # TODO: Not sure I like node2trackID

    def __get_common_track_IDs(
        self,
        new_edge: tuple[int, int],
        reference_edge: tuple[int, int] = None,
    ) -> set[int]:
        """
        Retrieves the common track IDs that are seen by the new edge and,
        if passed, the reference edge.

        Parameters
        ----------
        new_edge : tuple[int, int]
            New edge to process.
        reference_edge : tuple[int, int], optional
            Reference edge to use, by default None.

        Returns
        -------
        set[int]
            Common track IDs.
        """
        cameras = [*reference_edge, *new_edge] if reference_edge else [*new_edge]
        track_IDs = [self.camera2trackIDs[cam] for cam in set(cameras)]

        common_track_IDs = set(track_IDs[0])
        for lst in track_IDs[1:]:
            common_track_IDs.intersection_update(lst)
        return common_track_IDs

    def append_from_components(self, components: list[list[tuple[int, int]]], base_node: tuple[int, int]):
        """
        Generates list of Tracks from connected components and initial base node.

        Parameters
        ----------
        components : list[list[tuple[int, int]]]
            Connected components [[base_node, (cam_01, featureIdx), ...], [base_node, (cam_11, featureIdx), ...], ...]
            that represent the cameras that contain the same feature point (but may have different feature indices).
        base_node : tuple[int, int]
            Base (camera_id, featureIdx) pairing.
        """
        camera_id, _ = base_node

        for track in components:
            # Sort the track by camera_id if not already sorted and a set
            if isinstance(track, set):
                track = natsort.natsorted(list(track), key=lambda x: x[0])
            assert base_node == track[0], "First node should be the base node."
            # Convert to Track object
            track_ID = len(self)
            track_obj = Track(track_id=track_ID, track=track)
            self.append(track_obj)
            # Get the order of cameras in the track
            cameras_in_track = track_obj.cameras
            assert cameras_in_track[0] == camera_id, "First camera_id should be the base node's camera_id."
            # Add to camera2trackIDs
            for cam in cameras_in_track:
                self.camera2trackIDs[cam].append(track_ID)
            # Add to node2trackIDs
            for cam_featureIdx in track:
                self.node2trackIDs[cam_featureIdx].append(track_ID)

    def get_query_and_train(
        self,
        new_edge: tuple[int, int],
        reference_edge: tuple[int, int] = None,
    ) -> QueryTrainTracks:
        """
        Given the input edges, where an edge represents 2 cameras, will retrieve the
        track IDs that pass through, i.e, tracks that contain all the cameras desired.

        For these tracks, will retrieve the queryIdx, trainIdx, the 3D reference point
        and the reference inlier boolean.

        If new_edge is represented as (cam_i, cam_j) and reference edge is (cam_m, cam_j),
        then queryIdx and trainIdx are for cam_i and cam_j, respectively. And the 3D reference
        point and inlier flag is for reference edge.

        Parameters
        ----------
        new_edge : tuple[int, int]
            New edge to process, represents of a tuple of 2 cameras.
        camera2trackIDs : dict[int, list[int]]
            Mapping from camera_id to all tracks (via indices) that contain camera_id.
        is_initial_edge : bool, optional
            Whether the edge is the initial edge, by default True.
        reference_edge : tuple[int, int], optional
            Reference edge to use, represents of a tuple of 2 cameras, by default None.

        Returns
        -------
        QueryTrainTracks
            Query indices, train indices, 3D points, valid 3D points, and track objects.
            If is_initial_edge is False, then queryIdx and trainIdx will span all the edges passed.
        """
        # Get tracks that pass through the corresponding cameras
        common_track_IDs = self.__get_common_track_IDs(new_edge, reference_edge)

        # Get the query and train indices
        camera_0, camera_1 = new_edge
        indices = [
            (
                (track_obj := self[track_id]),  # Track object
                track_obj.cam2featureIdx[camera_0],  # Query
                track_obj.cam2featureIdx[camera_1],  # Train
                track_obj.get_pt_3D(reference_edge),  # 3D point for reference_edge in the same track
                track_obj.is_inlier(reference_edge),  # Is valid 3D point (essential matrix inlier and z > 0)
            )
            for track_id in common_track_IDs
        ]
        if len(indices) == 0:
            return QueryTrainTracks(np.array([]), np.array([]), np.array([]), np.array([]), [])

        track_objs, queryIdx, trainIdx, pts_3D, pts_3D_valid = zip(*indices)
        return QueryTrainTracks(
            queryIdx_tracks=np.asarray(queryIdx).squeeze(),
            trainIdx_tracks=np.asarray(trainIdx).squeeze(),
            pts_3D_ref=np.asarray(pts_3D).squeeze(),
            valid_pts_3D=np.asarray(pts_3D_valid).squeeze(),
            track_objs=track_objs,
        )

    def update_tracks(
        self,
        tracks_to_update: QueryTrainTracks,
        new_edge: tuple[int, int],
        pts_3D: np.ndarray,
        inliers: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        queryIdx_direct: np.ndarray | None,
        trainIdx_direct: np.ndarray | None,
        direct_update: bool = False,
    ):

        query = tracks_to_update.queryIdx_tracks
        train = tracks_to_update.trainIdx_tracks
        if direct_update:
            query = queryIdx_direct
            train = trainIdx_direct

        assert len(pts_3D) == len(inliers) == len(src_pts) == len(dst_pts) == len(query) == len(train)

        # Update track objects with triangulated info
        query2pts3D = dict(zip(query, pts_3D))
        query2srcpt = dict(zip(query, src_pts))
        query2valid = dict(zip(query, inliers))
        train2dstpt = dict(zip(train, dst_pts))
        for track_obj, queryIdx, trainIdx in zip(
            tracks_to_update.track_objs, tracks_to_update.queryIdx_tracks, tracks_to_update.trainIdx_tracks
        ):
            if direct_update and query2pts3D.get(queryIdx, None) is None:
                continue
            if direct_update and train2dstpt.get(trainIdx, None) is None:
                continue
            track_obj.update_triangulated(
                *new_edge,
                query2pts3D[queryIdx],
                query2valid[queryIdx],
                query2srcpt[queryIdx],
                train2dstpt[trainIdx],
            )


class CovisibilityGraph:
    def __init__(self, image_data: list[ImageData], method: str = "dijkstra"):

        # Updated in __setup(...)
        self.G_tracks = nx.DiGraph()
        self.G_covisibility = nx.DiGraph()

        # Updated in __obtain_tracks(...)
        self.tracks = Tracks()

        self.__setup(image_data)
        self.__obtain_tracks(method)

    def __setup(self, image_data: list[ImageData]):

        self.image_data = image_data
        pair_loader = get_image_matcher_loader(image_data)
        for idx, [(camera_0, camera_1, matches)] in enumerate(tqdm.tqdm(pair_loader, desc="Matching features")):
            self.add_to_tracks_graph(camera_0, camera_1, matches)
            self.add_to_covisibility_graph(camera_0, camera_1, matches)

    def __obtain_tracks(self, method: str = "dijkstra"):
        """
        Obtain the corresponding tracks of feature points. Results are similar
        to those of: https://imagine.enpc.fr/~moulonp/publis/poster_CVMP12.pdf.

        Parameters
        ----------
        method : str, optional
            Method to use for obtaining tracks, by default dijkstra.
        """
        start = time.time()
        base_nodes = [node for node in self.G_tracks.nodes if self.G_tracks.in_degree(node) == 0]

        for base_node in base_nodes:
            # Get connected components starting from a base node
            components = get_connected_components(self.G_tracks, base_node=base_node, method=method)
            self.tracks.append_from_components(components, base_node)

        end = time.time()
        logger.info(f"Number of base nodes: {len(base_nodes)}.")
        logger.info(f"Number of tracks obtained: {len(self.tracks)}.")
        logger.info(f"Track extraction time taken: {end - start} seconds.")

    def add_to_tracks_graph(self, camera_0: int, camera_1: int, matches: list[DMatch]):
        """
        Adds an edge to the graph for each matches. Each node will be a tuple of
        (camera_id, feature_id) and base=True if it is a base node else False.

        Parameters
        ----------
        camera_0 : int
            Camera 1 index.
        camera_1 : int
            Camera 2 index.
        matches : list[DMatch]
            Keypoint matches between 2 images.
        """
        for m in matches:
            src_node = (camera_0, m.queryIdx)  # (camera_0, featureIdx) for src feature
            dst_node = (camera_1, m.trainIdx)  # (camera_1, featureIdx) for dst feature

            # Add nodes between corresponding features
            if not self.G_tracks.has_node(src_node):
                self.G_tracks.add_node(node_for_adding=src_node)

            if not self.G_tracks.has_node(dst_node):
                self.G_tracks.add_node(node_for_adding=dst_node)

            # Add edge between corresponding features
            self.G_tracks.add_edge(u_of_edge=src_node, v_of_edge=dst_node)

    def add_to_covisibility_graph(self, camera_0: int, camera_1: int, matches: list[DMatch]):
        """
        Adds an edge to the graph for each matches. Each node will be a camera id
        with edges containing the matches and weight.

        Parameters
        ----------
        camera_0 : int
            Camera 1 index.
        camera_1 : int
            Camera 2 index.
        matches : list[DMatch]
            Keypoint matches between 2 images.
        """
        query2train = [(m.queryIdx, m.trainIdx) for m in matches]
        queryIdx, trainIdx = zip(*query2train)
        queryIdx = np.asarray(queryIdx).squeeze()
        trainIdx = np.asarray(trainIdx).squeeze()

        self.G_covisibility.add_node(node_for_adding=camera_0)
        self.G_covisibility.add_node(node_for_adding=camera_1)
        self.G_covisibility.add_edge(
            u_of_edge=camera_0,
            v_of_edge=camera_1,
            processed=False,  # Updated in get_next_edge(...)
            weight=len(matches),  # Used in get_next_edge(...)
            query_train=(queryIdx, trainIdx),
            num_pts_3D=0,  # Updated after triangulation
            stereo_camera=None,  # Updated after triangulation
            inliers=None,  # Updated after triangulation
        )

    def get_camera_span_from_edge(
        self,
        new_edge: tuple[int, int],
        reference_edge: tuple[int, int] = None,
    ) -> QueryTrainTracks | None:
        """
        Get corresponding tracks that span all cameras in reference_edge and new_edge. These
        tracks represent the union of source and destination points from all cameras seen between
        reference_edge and new_edge.

        For these tracks, will retrieve the queryIdx, trainIdx, the 3D reference point
        and the reference inlier boolean.

        It is possible that the cameras that span the track are not direct. For example, the track
        could connect from `cam_i ---> cam_j ---> cam_m ---> cameras_in_between ---> cam_n`.

        If new_edge is represented as (cam_i, cam_j) and reference edge is (cam_m, cam_n),
        then queryIdx and trainIdx are for cam_i and cam_j, respectively. And the 3D reference
        point and inlier flag is for reference edge.

        Parameters
        ----------
        new_edge : tuple[int, int]
            _description_
        reference_edge : tuple[int, int], optional
            _description_, by default None

        Returns
        -------
        QueryTrainTracks | None
            _description_
        """
        ref2new_tracks = self.tracks.get_query_and_train(new_edge=new_edge, reference_edge=reference_edge)
        if ref2new_tracks.queryIdx_tracks.size == 0:
            self.G_covisibility.add_edges_from([new_edge], num_pts_3D=-1)
            logger.info(f"Not enough query/train points for {new_edge}.")
            return None

        return ref2new_tracks

    def get_next_edge(self, init_edge: tuple[int, int]):
        """
        Get the next edge to process.

        Parameters
        ----------
        init_edge : tuple[int, int]
            Initial edge

        Returns
        -------
        tuple[int, int]
            reference_edge should be a pair of cameras that has already been processed and triangulated
            new_edge should be a new pair of cameras that succeed from reference_edge
        """

        def get_adjacent_edges(edge: tuple):
            """
            Get all edges adjacent to a given edge in a directed graph.

            Parameters
            ----------
            edge : tuple
                The edge (u, v).

            Returns
            -------
            set
                Set of edges adjacent to the given edge.
            """
            G = self.G_covisibility
            u, v = edge
            adjacent_edges = set(G.out_edges(v)) | set(G.in_edges(u)) | set(G.out_edges(u)) | set(G.in_edges(v))
            return adjacent_edges

        # Priority queue: (-weight, -num_pts, reference_edge, edge)
        heap: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []

        # Initialize with the init_edge
        for edge in get_adjacent_edges(init_edge):
            if not self.G_covisibility.edges[edge]["processed"]:
                weight = self.G_covisibility.edges[edge]["weight"]
                num_pts = self.G_covisibility.edges[init_edge]["num_pts_3D"]
                # Use negative values because heapq uses min-heap
                heapq.heappush(heap, (-num_pts, -weight, init_edge, edge))

        while heap:
            # Get the edge with the highest weight and num_pts
            ref_num_pts, ref_weight, reference_edge, new_edge = heapq.heappop(heap)
            if not self.G_covisibility.edges[new_edge]["processed"]:
                self.G_covisibility.edges[new_edge]["processed"] = True
                yield ref_num_pts, ref_weight, reference_edge, new_edge

                # Add adjacent edges to the heap
                for adjacent_edge in get_adjacent_edges(new_edge):
                    if not self.G_covisibility.edges[adjacent_edge]["processed"]:
                        weight = self.G_covisibility.edges[adjacent_edge]["weight"]
                        num_pts = self.G_covisibility.edges[new_edge]["num_pts_3D"]
                        heapq.heappush(heap, (-num_pts, -weight, new_edge, adjacent_edge))


def unique_dfs(graph: nx.DiGraph, base_node: tuple[int, int], *args) -> list[list[tuple[int, int]]]:
    """
    Retrieves the unique connected component(s) of a graph starting from a base node.
    When forks are detected, will split and copy the component for the new path(s).

    Parameters
    ----------
    graph : nx.DiGraph
        Graph.
    base_node : tuple[int, int]
        Base not to start from.

    Returns
    -------
    list[list[tuple[int, int]]]
        List of tracks.
    """

    def dfs(node, current_path: list, all_paths: list):
        current_path.append(node)
        successors = list(graph.successors(node))

        if not successors:
            # If no successors, we've reached the end of a path
            all_paths.append(current_path.copy())
        else:
            for successor in successors:
                dfs(successor, current_path, all_paths)

        current_path.pop()  # Backtrack

    starting_nodes = [base_node]  # [node for node in graph.nodes if graph.in_degree(node) == 0]
    all_paths = []
    for start_node in starting_nodes:
        dfs(start_node, [], all_paths)

    return all_paths


def get_connected_components(
    G: nx.DiGraph, base_node: tuple[int, int], method: str = "dfs"
) -> list[list[tuple[int, int]]]:
    """
    Retrieves the connected component(s) of a graph starting from a base node.

    Parameters
    ----------
    G : nx.DiGraph
        Graph.
    base_node : tuple[int, int]
        Base not to start from.
    method : bool, optional
        Method to use for track extraction, by default False.

    Returns
    -------
    list[list[tuple[int, int]]]
        List of tracks.
    """
    if method == "dijkstra":
        count, path = nx.single_source_dijkstra(G, source=base_node)

        max_value = max(count.values())
        last_node = [key for key, value in count.items() if value == max_value]
        component = [path[key] for key in last_node]
    elif method == "dfs":
        # NOTE: Edges that join to a node work fine, but edges that fork from a node are a little tricky.
        component = nx.kosaraju_strongly_connected_components(G, base_node)
    else:
        component = unique_dfs(G, base_node)
    return component

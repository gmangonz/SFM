import heapq
import time
from collections import defaultdict
from dataclasses import dataclass

import natsort
import networkx as nx
import numpy as np

from sfm.image_matching import DMatch
from sfm.utils import ImageData, logger


@dataclass
class Track:
    """In theory, represents a single (3D) point that propagates from an initial image to subsequent images."""

    track_id: int
    track: list[tuple[int, int]] | np.ndarray
    valid_pts: int = 0

    def __post_init__(self):

        # Mapping from camera_id to feature index
        self.cam2featureIdx = dict(self.track)
        # Get the cameras in order seen in the track
        self.cameras = np.asarray(self.track)[:, 0]
        # Initialize mapping from edge to triangulated boolean, 3D point, inlier status, and respective 2D points
        self._edge2pt = defaultdict(lambda: np.array([-1.0, -1.0, -1.0]))
        self._edge2inlier = defaultdict(lambda: False)
        self._edge2src = defaultdict(lambda: np.array([-1.0, -1.0]))
        self._edge2dst = defaultdict(lambda: np.array([-1.0, -1.0]))

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
        if is_inlier and not _edge in self._edge2pt:
            self.valid_pts += 1

        self._edge2inlier[_edge] = is_inlier
        self._edge2pt[_edge] = pt_3D
        self._edge2src[_edge] = src
        self._edge2dst[_edge] = dst

    def get_pt_3D(self, edge: tuple[int, int] = None) -> np.ndarray:
        return np.array([-1.0, -1.0, -1.0]) if edge is None else self._edge2pt[edge]

    def is_inlier(self, edge: tuple[int, int] = None) -> bool:
        return False if edge is None else self._edge2inlier[edge]

    def is_valid_track(self, n: int = 3) -> bool:
        return self.valid_pts >= n

    def get_edges(self) -> list[tuple[int, int]]:
        return list(self._edge2pt.keys())

    def get_edge(self, edge) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, bool]:
        return (
            (self._edge2src[edge], self._edge2dst[edge]),
            self._edge2pt[edge],
            self._edge2inlier[edge],
        )

    def __len__(self) -> int:
        return len(self.track)


@dataclass
class QueryTrainTracks:
    queryIdx_tracks: np.ndarray
    trainIdx_tracks: np.ndarray
    pts_3D_ref: np.ndarray
    valid_pts_3D: np.ndarray
    track_objs: list[Track]


def get_edge_with_largest_weight(G: nx.DiGraph | nx.Graph) -> tuple[int, int]:
    """
    Finds the edge with the largest weight in a NetworkX graph.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Graph.

    Returns
    -------
    tuple[int, int]
        A tuple (u, v) representing the edge with the largest weight, or None if the graph is empty.
    """
    if not G.edges:
        return False

    max_weight = 0
    max_edge = False

    for u, v, data in G.edges(data=True):
        if "weight" in data and data["weight"] > max_weight:
            max_weight = data["weight"]
            max_edge = (u, v)
    return max_edge


def get_next_edge(G: nx.DiGraph, init_edge: tuple[int, int]):
    """
    Get the next edge to process.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Graph.

    Returns
    -------
    tuple[int, int]
        reference_edge should be a pair of cameras that has already been processed and triangulated
        new_edge should be a new pair of cameras that succeed from reference_edge
    """
    # Priority queue: (-weight, -num_pts, reference_edge, edge)
    heap: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []

    # Initialize with the init_edge
    for edge in get_adjacent_edges(G, init_edge):
        if not G.edges[edge]["processed"]:
            weight = G.edges[edge]["weight"]
            num_pts = G.edges[init_edge]["num_pts_3D"]
            # Use negative values because heapq uses min-heap
            heapq.heappush(heap, (-num_pts, -weight, init_edge, edge))

    while heap:
        # Get the edge with the highest weight and num_pts
        ref_num_pts, ref_weight, reference_edge, new_edge = heapq.heappop(heap)
        if not G.edges[new_edge]["processed"]:
            G.edges[new_edge]["processed"] = True
            yield ref_num_pts, ref_weight, reference_edge, new_edge

            # Add adjacent edges to the heap
            for adjacent_edge in get_adjacent_edges(G, new_edge):
                if not G.edges[adjacent_edge]["processed"]:
                    weight = G.edges[adjacent_edge]["weight"]
                    num_pts = G.edges[new_edge]["num_pts_3D"]
                    heapq.heappush(heap, (-num_pts, -weight, new_edge, adjacent_edge))


def get_adjacent_edges(G: nx.DiGraph, edge: tuple):
    """
    Get all edges adjacent to a given edge in a directed graph.

    Parameters
    ----------
    G : nx.DiGraph
        The directed graph.
    edge : tuple
        The edge (u, v).

    Returns
    -------
    set
        Set of edges adjacent to the given edge.
    """
    u, v = edge
    adjacent_edges = set(G.out_edges(v)) | set(G.in_edges(u)) | set(G.out_edges(u)) | set(G.in_edges(v))
    return adjacent_edges


def retrieve_edge_data(G: nx.DiGraph | nx.Graph, edge: tuple[int, int]) -> dict:
    """
    Retrieves the edge data from a graph.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Graph.
    edge : tuple[int, int]
        Tuple representing the edge.

    Returns
    -------
    dict
        Dictionary containing the edge data and the camera_0 and camera_1 indices.
    """
    return dict(**G.get_edge_data(*edge), camera_0=edge[0], camera_1=edge[1])


def add_to_covisibility_graph(G: nx.DiGraph, camera_0: int, camera_1: int, matches: list[DMatch]) -> nx.Graph:
    """
    Adds an edge to the graph for each matches. Each node will be a camera id
    with edges containing the matches and weight.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Graph.
    camera_0 : int
        Camera 1 index.
    camera_1 : int
        Camera 2 index.
    matches : list[DMatch]
        Keypoint matches between 2 images.

    Returns
    -------
    nx.Graph
        Updated graph.
    """
    query2train = [(m.queryIdx, m.trainIdx) for m in matches]
    queryIdx, trainIdx = zip(*query2train)
    queryIdx = np.asarray(queryIdx).squeeze()
    trainIdx = np.asarray(trainIdx).squeeze()

    G.add_node(node_for_adding=camera_0)
    G.add_node(node_for_adding=camera_1)
    G.add_edge(
        u_of_edge=camera_0,
        v_of_edge=camera_1,
        processed=False,  # Updated in get_next_edge(...)
        weight=len(matches),  # Used in get_next_edge(...)
        query_train=(queryIdx, trainIdx),
        num_pts_3D=0,  # Updated after triangulation
        stereo_camera=None,  # Updated after triangulation
        inliers=None,  # Updated after triangulation
    )
    return G


def add_to_tracks_graph(G: nx.DiGraph, camera_0: int, camera_1: int, matches: list[DMatch]) -> nx.DiGraph:
    """
    Adds an edge to the graph for each matches. Each node will be a tuple of
    (camera_id, feature_id) and base=True if it is a base node else False.

    Parameters
    ----------
    G : nx.DiGraph | nx.Graph
        Graph.
    camera_0 : int
        Camera 1 index.
    camera_1 : int
        Camera 2 index.
    matches : list[DMatch]
        Keypoint matches between 2 images.

    Returns
    -------
    nx.DiGraph
        Updated graph.
    """
    for m in matches:
        src_node = (camera_0, m.queryIdx)  # (camera_0, featureIdx) for src feature
        dst_node = (camera_1, m.trainIdx)  # (camera_1, featureIdx) for dst feature

        # Add nodes between corresponding features
        if not G.has_node(src_node):
            G.add_node(node_for_adding=src_node)

        if not G.has_node(dst_node):
            G.add_node(node_for_adding=dst_node)

        # Add edge between corresponding features
        G.add_edge(u_of_edge=src_node, v_of_edge=dst_node)
    return G


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


def _get_component(G: nx.DiGraph, base_node: tuple[int, int], method: str = "dfs") -> list[list[tuple[int, int]]]:
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


def obtrain_tracks(
    G: nx.DiGraph, method: str = "dijkstra"
) -> tuple[list[Track], dict[int, list[int]], dict[tuple[int, int], list[int]]]:
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
    start = time.time()
    camera2trackID: dict[int, list[int]] = defaultdict(list)
    tracks: list[Track] = []
    node2trackID: dict[tuple[int, int], list[int]] = defaultdict(list)

    base_nodes = [node for node in G.nodes if G.in_degree(node) == 0]

    for base_node in base_nodes:
        # Get connected components starting from a base node
        component = _get_component(G, base_node, method=method)
        camera_id, _ = base_node

        for track in component:
            if len(track) < 3:
                continue
            # Sort the track by camera_id if not already sorted and a set
            if isinstance(track, set):
                track = natsort.natsorted(list(track), key=lambda x: x[0])
            assert base_node == track[0], "First node should be the base node."
            # Convert to Track object
            track_ID = len(tracks)
            track_obj = Track(track_id=track_ID, track=track)
            tracks.append(track_obj)
            # Get the order of cameras in the track
            cameras_in_track = track_obj.cameras
            assert cameras_in_track[0] == camera_id, "First camera_id should be the base node's camera_id."
            # Add to camera2trackID
            for cam in cameras_in_track:
                camera2trackID[cam].append(track_ID)
            # Add to node2trackID
            for cam_featureIdx in track:
                node2trackID[cam_featureIdx].append(track_ID)

    end = time.time()
    logger.info(f"Number of base nodes: {len(base_nodes)}.")
    logger.info(f"Number of tracks obtained: {len(tracks)}.")
    logger.info(f"Track extraction time taken: {end - start} seconds.")
    return tracks, camera2trackID, node2trackID


def get_common_track_IDs(
    new_edge: tuple[int, int],
    camera2trackIDs: dict[int, list[int]],
    reference_edge: tuple[int, int] = None,
) -> set[int]:
    """
    Retrieves the common track IDs that are seen by the new edge and,
    if passed, the reference edge.

    Parameters
    ----------
    new_edge : tuple[int, int]
        New edge to process.
    camera2trackIDs : dict[int, list[int]]
        Mapping from camera_id to all tracks (via indices) that contain camera_id.
    reference_edge : tuple[int, int], optional
        Reference edge to use, by default None.

    Returns
    -------
    set[int]
        Common track IDs.
    """

    cameras = [*reference_edge, *new_edge] if reference_edge else [*new_edge]
    track_IDs = [camera2trackIDs[cam] for cam in cameras]
    common_track_IDs = set(track_IDs[0])
    for lst in track_IDs[1:]:
        common_track_IDs.intersection_update(lst)
    return common_track_IDs


def get_query_and_train(
    new_edge: tuple[int, int],
    tracks: list[Track],
    camera2trackIDs: dict[int, list[int]],
    is_initial_edge: bool = True,
    reference_edge: tuple[int, int] = None,
) -> QueryTrainTracks:
    """
    Given the input edges, where an edge represents 2 cameras, will retrieve the
    track IDs that pass through, i.e, tracks that contain all the cameras desired.

    For these tracks, will retrieve the queryIdx, trainIdx, the 3D reference point
    and the reference inlier boolean.

    Parameters
    ----------
    new_edge : tuple[int, int]
        New edge to process, represents of a tuple of 2 cameras.
    tracks : list[Track]
        List of tracks, where each track is a sorted list of tuples
        (camera_id, feature_id) by camera_id.
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
    if not is_initial_edge and not reference_edge:
        raise ValueError("reference_edge must be passed if is_initial_edge is False.")

    # Get tracks that pass through the corresponding cameras
    common_track_IDs = get_common_track_IDs(new_edge, camera2trackIDs, reference_edge)

    # Get the query and train indices
    camera_0, camera_1 = new_edge
    indices = [
        (
            (track_obj := tracks[track_id]),  # Track object
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

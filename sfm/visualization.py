import os

import cv2
import gtsam
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from sfm.utils import ImageData, extract_camera_poses, to_cv2KeyPoint

__cwd__ = os.path.dirname(os.path.abspath(__file__))


def plot_matches(
    image_i: ImageData, image_j: ImageData, queryIdx: np.ndarray, trainIdx: np.ndarray, inliers: np.ndarray
) -> None:

    edge = "-".join([image_i.filename, image_j.filename])

    print(f"Plotting matches for {image_i.filename} and {image_j.filename}")
    image_1, clahe_img_1, kpts_1, _ = image_i.image, image_i.clahe_img, image_i.keypoints, image_i.features
    image_2, clahe_img_2, kpts_2, _ = image_j.image, image_j.clahe_img, image_j.keypoints, image_j.features

    kpts_1 = [to_cv2KeyPoint(kpt) for kpt in kpts_1]
    kpts_2 = [to_cv2KeyPoint(kpt) for kpt in kpts_2]

    image_1 = (image_1 * 255).astype(np.uint8)
    image_2 = (image_2 * 255).astype(np.uint8)

    out_1 = cv2.drawKeypoints(image_1.copy(), kpts_1, image_1.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_2 = cv2.drawKeypoints(image_2.copy(), kpts_2, image_2.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    query: list[int] = queryIdx.tolist()
    train: list[int] = trainIdx.tolist()

    src_pts = np.float32([kpts_1[m].pt for m in query]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_2[m].pt for m in train]).reshape(-1, 1, 2)

    concat_image = np.concatenate((image_1, image_2), axis=1)
    match_img = concat_image.copy()
    offset = concat_image.shape[1] / 2

    ASPECT = None  # "equal"
    fig = plt.figure(figsize=(10, 14))
    plt.subplots_adjust(hspace=0.05)

    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[0.5, 0.5])  # 4 rows, 2 columns

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(clahe_img_1, cmap="gray", aspect=ASPECT)
    ax.set_title("Image 1")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(out_1, aspect=ASPECT)
    ax.set_title("Original Image 1 with SIFT features")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(clahe_img_2, cmap="gray", aspect=ASPECT)
    ax.set_title("Image 2")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(out_2, aspect=ASPECT)
    ax.set_title("Original Image 2 with SIFT features")
    ax.axis("off")

    ax = fig.add_subplot(gs[2, :])
    ax.imshow(np.array(match_img).astype(np.uint8), aspect=ASPECT)  # RGB is integer type
    ax.plot(src_pts.squeeze()[:, 0], src_pts.squeeze()[:, 1], "xr")
    ax.plot(dst_pts.squeeze()[:, 0] + offset, dst_pts.squeeze()[:, 1], "xr")
    ax.plot(
        [src_pts.squeeze()[:, 0], dst_pts.squeeze()[:, 0] + offset],
        [src_pts.squeeze()[:, 1], dst_pts.squeeze()[:, 1]],
        "r",
        linewidth=0.5,
    )
    ax.set_title(f"Matches before RANSAC, n={len(src_pts)}")
    ax.axis("off")

    src_pts = src_pts.squeeze()[inliers]
    dst_pts = dst_pts.squeeze()[inliers]

    ax = fig.add_subplot(gs[3, :])
    ax.imshow(np.array(match_img).astype(np.uint8), aspect=ASPECT)  # RGB is integer type
    ax.plot(src_pts[:, 0], src_pts[:, 1], "xr")
    ax.plot(dst_pts[:, 0] + offset, dst_pts[:, 1], "xr")
    ax.plot(
        [src_pts[:, 0], dst_pts[:, 0] + offset],
        [src_pts[:, 1], dst_pts[:, 1]],
        "r",
        linewidth=0.5,
    )
    ax.set_title(f"Matches after RANSAC, n={sum(inliers)}")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(__cwd__, "debug", f"matches_{edge}.png"))
    plt.close()


def visualize_graph(
    estimates,
    colors: dict[gtsam.Symbol, np.ndarray],
    pt_scale: int = 1,
    title: str = "Structure from Motion",
):

    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
        ),
        title=title,
    )

    landmarks, pt_colors, camera_trans, camera_names, frustums = extract_camera_poses(estimates, colors)
    landmarks = np.asarray(landmarks)
    pt_colors = np.asarray(pt_colors) / 255.0

    landmark_scatter = go.Scatter3d(
        x=landmarks[:, 0],
        y=landmarks[:, 1],
        z=landmarks[:, 2],
        mode="markers",
        name="Landmarks",
        marker=dict(size=pt_scale, color=pt_colors),
    )

    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    for name, vertices, t in zip(camera_names, frustums, camera_trans):
        camera_scatter = go.Scatter3d(
            x=[t[0]],
            y=[t[1]],
            z=[t[2]],
            mode="markers",
            name=name,
            legendgroup="Cameras",
            showlegend=False,
            marker=dict(size=3, color="blue"),
        )
        fig.add_trace(camera_scatter)

        x, y, z = vertices.T
        pyramid = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color="rgba(255,0,0,0.5)",
            i=i,
            j=j,
            k=k,
            legendgroup="Cameras",
            name=name,
            showlegend=False,
        )
        fig.add_trace(pyramid)

        triangles = np.vstack((i, j, k)).T
        triangles = np.array([vertices[i] for i in triangles.reshape(-1)])
        x, y, z = triangles.T

        pyramid = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            legendgroup="Frustums",
            name=name,
            showlegend=False,
            line=dict(color="rgba(255,0,0,0.5)", width=1),
        )
        fig.add_trace(pyramid)

    fig.add_traces(data=[landmark_scatter])
    fig.show()

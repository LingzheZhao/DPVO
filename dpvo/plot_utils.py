from itertools import chain
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from plyfile import PlyData, PlyElement

import dpvo.colmap_utils as colmap_utils


def plot_trajectory(
    pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True
):
    assert isinstance(pred_traj, PoseTrajectory3D)

    if gt_traj is not None:
        assert isinstance(gt_traj, PoseTrajectory3D)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xz  # ideal for planar movement
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")


# Ensure save directories exist
def init_file_structure(save_path: Path):
    save_path.mkdir(exist_ok=True, parents=True)
    images_path = save_path / "images"
    sparse_path = save_path / "sparse/0"
    images_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)
    return save_path, images_path, sparse_path


def save_output_for_COLMAP(
    output_dir: Path,
    traj: PoseTrajectory3D,
    points: np.ndarray,
    colors: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H: int,
    W: int,
    imagedir: Path,
):
    """Saves the sparse point cloud and camera poses such that it can be opened in COLMAP"""

    save_path, images_save_dir, sparse_save_dir = init_file_structure(output_dir)

    # w2c to c2w
    traj = PoseTrajectory3D(
        poses_se3=list(map(np.linalg.inv, traj.poses_se3)), timestamps=traj.timestamps
    )

    # Save images
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    images_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))

    if not len(images_list) == traj.num_poses:
        print(
            f"Warning: Number of images ({len(images_list)}) does not match number of poses ({traj.num_poses})"
        )

    colmap_utils.save_images(images_save_dir, images_list)

    # Save cameras
    world2cam = np.stack(traj.poses_se3, axis=0)
    focals = np.array([[fx, fy]])
    principal_points = np.array([[cx, cy]])
    img0 = cv2.imread(str(images_list[0]))
    imgs_shape = (len(images_list), img0.shape[0], img0.shape[1])
    colmap_utils.save_cameras(
        focals, principal_points, sparse_save_dir, imgs_shape=imgs_shape
    )
    if not world2cam.shape[0] == len(images_list):
        print(
            f"Warning: Number of poses ({world2cam.shape[0]}) does not match number of images ({len(images_list)})"
        )
    colmap_utils.save_images_txt(world2cam, images_list, sparse_save_dir)
    camera_entities = colmap_utils.read_cameras_text(sparse_save_dir / "cameras.txt")
    colmap_utils.write_cameras_binary(camera_entities, sparse_save_dir / "cameras.bin")
    images_entities = colmap_utils.read_images_text(sparse_save_dir / "images.txt")
    colmap_utils.write_images_binary(images_entities, sparse_save_dir / "images.bin")

    # Save pointcloud
    colors, vertices = colmap_utils.save_pointcloud_with_normals(
        sparse_save_dir,
        points,
        colors=colors,
        imgs=None,
    )
    points3d = {}
    for i in range(len(points)):
        points3d[i] = colmap_utils.Point3D(
            id=i,
            xyz=vertices[i],
            rgb=colors[i][:3],
            error=np.array([0]),
            image_ids=np.array([0]),
            point2D_idxs=np.array([0]),
        )
    pcl_bin_save_path = sparse_save_dir / "points3D.bin"
    colmap_utils.write_points3D_binary(points3d, pcl_bin_save_path)


def save_ply(output_dir: Path, points: np.ndarray, colors: np.ndarray):
    _, _, sparse_save_dir = init_file_structure(output_dir)
    points_ply = np.array(
        [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    el = PlyElement.describe(
        points_ply, "vertex", {"some_property": "f8"}, {"some_property": "u4"}
    )
    PlyData([el], text=True).write(sparse_save_dir / "points3D_dpvo_original.ply")
    print(f"Saved {sparse_save_dir / 'points3D_dpvo_original.ply'}")

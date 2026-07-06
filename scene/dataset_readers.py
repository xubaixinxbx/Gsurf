#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    return {"translate": -center, "radius": diagonal * 1.1}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(path)


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, intr.height)
            FovX = focal2fov(focal_length_x, intr.width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, intr.height)
            FovX = focal2fov(focal_length_x, intr.width)
        else:
            raise ValueError(
                "Unsupported COLMAP camera model. Use undistorted PINHOLE or SIMPLE_PINHOLE cameras."
            )

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=intr.width,
                height=intr.height,
            )
        )
    sys.stdout.write("\n")
    return cam_infos


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting COLMAP points3D to points3D.ply.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    return SceneInfo(
        point_cloud=fetchPly(ply_path),
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


def _resolve_frame_image_path(root, prefix, file_path, extension):
    rel = Path(file_path)
    if not rel.suffix:
        rel = Path(str(rel) + extension)
    if rel.is_absolute():
        return str(rel)
    if prefix and rel.parts and rel.parts[0] == prefix:
        return str(Path(root) / rel)
    return str(Path(root) / prefix / rel)


def readCamerasFromTransformsOmni(path, transformsfile, white_background, extension=".png", prefix="images"):
    del white_background
    cam_infos = []
    with open(os.path.join(path, transformsfile), "r", encoding="utf-8") as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]
    for idx, frame in enumerate(contents["frames"]):
        image_path = _resolve_frame_image_path(path, prefix, frame["file_path"], extension)
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=fovy,
                FovX=fovx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos


def _depth_path_for_image(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    try:
        image_idx = parts.index("images")
        parts[image_idx] = "depth"
        return Path(*parts)
    except ValueError:
        return image_path.parent.parent / "depth" / image_path.name


def _seed_points_from_depth(cam_infos, samples_per_view=100000):
    import torch

    def depths_to_points(world_view_transform, W, H, fov, depthmap):
        c2w = world_view_transform.T.inverse()
        fx = W / (2 * np.tan(fov / 2.0))
        fy = H / (2 * np.tan(fov / 2.0))
        intrins = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            device="cuda",
        ).float()
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device="cuda").float() + 0.5,
            torch.arange(H, device="cuda").float() + 0.5,
            indexing="xy",
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        return depthmap.reshape(-1, 1) * rays_d + rays_o

    xyz = []
    for cam_info in cam_infos:
        depth_img_path = _depth_path_for_image(cam_info.image_path)
        if not depth_img_path.exists():
            raise FileNotFoundError(f"Depth map not found for {cam_info.image_path}: {depth_img_path}")

        world2view = torch.tensor(getWorld2View2(cam_info.R, cam_info.T), device="cuda").transpose(0, 1)
        depth_img = Image.open(depth_img_path).convert("L")
        depthmap = torch.from_numpy(np.array(depth_img) / 255.0 * 100.0).float().cuda()
        points = depths_to_points(world2view, cam_info.width, cam_info.height, cam_info.FovX, depthmap)
        sample_count = min(int(samples_per_view), points.shape[0])
        indices = torch.randperm(points.shape[0], device=points.device)[:sample_count]
        xyz.append(points[indices].cpu().numpy())

    return np.concatenate(xyz, axis=0)


def _random_seed_points(num_pts=100000):
    return np.random.random((num_pts, 3)) * 2.6 - 1.3


def readOmniInfo(path, white_background, eval, extension=".png", llffhold=8):
    print("Reading OmniObject3D transforms")
    all_cam_infos = readCamerasFromTransformsOmni(
        path,
        "transforms.json",
        white_background,
        extension,
        prefix="images",
    )

    if eval:
        train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = all_cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")

    if not os.path.exists(ply_path):
        depth_map_path = os.path.join(path, "depth")
        if os.path.exists(depth_map_path):
            print("Initializing OmniObject3D point cloud from depth maps.")
            xyz = _seed_points_from_depth(all_cam_infos)
        else:
            print("Depth maps not found; initializing OmniObject3D point cloud randomly.")
            xyz = _random_seed_points()

        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    return SceneInfo(
        point_cloud=fetchPly(ply_path),
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Omni": readOmniInfo,
}

# GSurf

Official code for **GSurf: Learning signed distance fields from splatting opaque Gaussians for high-quality 3D reconstruction**.

[Paper](https://arxiv.org/pdf/2411.15723) | [Pretrained meshes](https://drive.google.com/drive/folders/1bikta-AmHjA-JjCI3RGbSHRRQrOrTw4L?usp=sharing)



https://github.com/user-attachments/assets/6132c494-8cf0-4791-9d51-435fdcd761f5



This repository contains the training and inference code path used for GSurf on:

- DTU scenes preprocessed in the 2DGS/3DGS COLMAP layout (`sparse/0`, `images/`)
- OmniObject3D scenes with `transforms.json` and `images/`


## Installation

```bash
conda env create -f environment.yml
conda activate gsurf
```

The environment file installs the two CUDA extensions from the source folders:

```bash
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

If you already have a compatible PyTorch/CUDA environment, installing the two extensions above plus the Python packages in `environment.yml` is sufficient.

## Data Layout

DTU:

```text
<dtu_scene>/
  images/
  sparse/0/
    cameras.bin or cameras.txt
    images.bin or images.txt
    points3D.bin or points3D.txt
```

OmniObject3D:

```text
<omni_scene>/
  transforms.json
  images/
  depth/          # optional; used only to initialize points3d.ply
```

For OmniObject3D, the loader creates `points3d.ply` on first use. If `depth/` exists, it initializes from depth maps; otherwise it uses random points.

## Training

DTU:

```bash
python train.py -s /path/to/dtu/scan105 -m output/dtu/scan105 -r 2 --depth_ratio 1 --gsurf
```

OmniObject3D:

```bash
python train.py -s /path/to/omni/object_scene -m output/omni/object_scene --gsurf
```

Useful GSurf options:

```bash
--gsurf
--gsurf_sphere_radius 0.6
--gsurf_sdf_lr 0.0001
--gsurf_rgb_lr 0.0001
--gsurf_loss_pos 0.1
--gsurf_loss_eik 0.01
--gsurf_loss_off 0.01
--gsurf_loss_ori 0.05
--gsurf_loss_normal 0.05
--gsurf_loss_entropy 0.01
```

Checkpoints and point clouds are saved under:

```text
<model_path>/point_cloud/iteration_<iter>/
  point_cloud.ply
  gsurf_networks.pth
```

## Rendering and Mesh Extraction

Render train/test images and extract a TSDF mesh:

```bash
python render.py -s /path/to/dtu/scan105 -m output/dtu/scan105 -r 2 --depth_ratio 1 --gsurf --skip_train --skip_test
```

Extract a mesh directly from the learned GSurf SDF:

```bash
python render.py -s /path/to/dtu/scan105 -m output/dtu/scan105 --gsurf_mesh --gsurf_mesh_res 256 --skip_train --skip_test
```

The mesh is written to:

```text
<model_path>/train/ours_<iter>/fuse.ply
<model_path>/train/ours_<iter>/fuse_post.ply
```

## Acknowledgements

This code builds on 2D Gaussian Splatting and the 3D Gaussian Splatting codebase. The surfel rasterizer and simple-knn CUDA extensions are vendored in `submodules/` for release convenience.

## Citation
If you find our code or paper useful, please consider citing
```
@article{XU2026gsurf,
title = {GSurf: Learning signed distance fields from splatting opaque Gaussians for high-quality 3D reconstruction},
journal = {Computer-Aided Design},
volume = {199},
pages = {104106},
year = {2026},
issn = {0010-4485},
doi = {https://doi.org/10.1016/j.cad.2026.104106},
url = {https://www.sciencedirect.com/science/article/pii/S001044852600076X},
author = {Baixin Xu and Jiangbei Hu and Jiaze Li and Ying He},
}
```

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import build_rotation


class ProgressivePositionalEncoding(nn.Module):
    def __init__(
        self,
        input_dims=3,
        num_freqs=6,
        start_iter=0,
        end_iter=5000,
        include_input=True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = int(num_freqs)
        self.start_iter = int(start_iter)
        self.end_iter = int(end_iter)
        self.include_input = include_input
        freq_bands = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("cur_iteration", torch.zeros((), dtype=torch.float32), persistent=False)
        self.out_dim = (input_dims if include_input else 0) + 2 * self.num_freqs * input_dims

    def set_iteration(self, iteration):
        self.cur_iteration.fill_(float(iteration))

    def _freq_weights(self):
        if self.num_freqs == 0:
            return self.freq_bands.new_zeros((0,))
        if self.end_iter <= self.start_iter:
            return self.freq_bands.new_ones((self.num_freqs,))

        progress = (self.cur_iteration - float(self.start_iter)) / float(self.end_iter - self.start_iter)
        progress = progress.clamp(0.0, 1.0) * self.num_freqs
        bands = torch.arange(self.num_freqs, device=self.freq_bands.device, dtype=self.freq_bands.dtype)
        weights = (progress - bands).clamp(0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(math.pi * weights))

    def forward(self, inputs):
        encoded = []
        if self.include_input:
            encoded.append(inputs)
        if self.num_freqs > 0:
            scaled = inputs[..., None, :] * self.freq_bands[:, None] * math.pi
            pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
            pe = pe * self._freq_weights()[:, None]
            encoded.append(pe.reshape(*inputs.shape[:-1], -1))
        return torch.cat(encoded, dim=-1)


class GSurfSDFNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_layers=8,
        geo_feat_dim=64,
        skip_in=(4,),
        sphere_radius=0.6,
        pe_freqs=6,
        pe_start_iter=500,
        pe_end_iter=5000,
    ):
        super().__init__()
        self.geo_feat_dim = geo_feat_dim
        self.skip_in = set(skip_in)
        self.sphere_radius = sphere_radius
        self.encoder = ProgressivePositionalEncoding(
            input_dims=3,
            num_freqs=pe_freqs,
            start_iter=pe_start_iter,
            end_iter=pe_end_iter,
            include_input=True,
        )

        in_dim = self.encoder.out_dim
        layers = []
        for layer_idx in range(num_layers):
            layer_in_dim = hidden_dim
            if layer_idx == 0:
                layer_in_dim = in_dim
            elif layer_idx in self.skip_in:
                layer_in_dim = hidden_dim + in_dim
            linear = nn.Linear(layer_in_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Softplus(beta=100)
        self.sdf_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, geo_feat_dim)
        nn.init.normal_(self.sdf_head.weight, mean=np.sqrt(np.pi) / np.sqrt(hidden_dim), std=1e-4)
        nn.init.constant_(self.sdf_head.bias, -sphere_radius)
        nn.init.xavier_uniform_(self.feature_head.weight)
        nn.init.zeros_(self.feature_head.bias)

    def set_iteration(self, iteration):
        self.encoder.set_iteration(iteration)

    def forward(self, points):
        inputs = self.encoder(points)
        x = inputs
        for layer_idx, linear in enumerate(self.layers):
            if layer_idx in self.skip_in:
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2.0)
            x = self.activation(linear(x))
        sdf = self.sdf_head(x)
        geo_features = self.feature_head(x)
        return sdf, geo_features

    def sdf(self, points):
        return self.forward(points)[0]

    def gradient(self, points, create_graph=True):
        points = points.requires_grad_(True)
        sdf = self.sdf(points)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients


class GSurfAppearanceNetwork(nn.Module):
    def __init__(self, geo_feat_dim=64, hidden_dim=256, num_layers=4):
        super().__init__()
        dims = [3 + 3 + 3 + geo_feat_dim] + [hidden_dim] * (num_layers - 1) + [3]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward(self, points, viewdirs, normals, geo_features):
        x = torch.cat([points, viewdirs, normals, geo_features], dim=-1)
        for linear in self.layers[:-1]:
            x = F.relu(linear(x), inplace=True)
        return torch.sigmoid(self.layers[-1](x))


class GSurfModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        sdf_layers=8,
        rgb_layers=4,
        geo_feat_dim=64,
        sphere_radius=0.6,
        pe_freqs=6,
        pe_start_iter=500,
        pe_end_iter=5000,
    ):
        super().__init__()
        self.config = {
            "hidden_dim": hidden_dim,
            "sdf_layers": sdf_layers,
            "rgb_layers": rgb_layers,
            "geo_feat_dim": geo_feat_dim,
            "sphere_radius": sphere_radius,
            "pe_freqs": pe_freqs,
            "pe_start_iter": pe_start_iter,
            "pe_end_iter": pe_end_iter,
        }
        self.sdf_network = GSurfSDFNetwork(
            hidden_dim=hidden_dim,
            num_layers=sdf_layers,
            geo_feat_dim=geo_feat_dim,
            sphere_radius=sphere_radius,
            pe_freqs=pe_freqs,
            pe_start_iter=pe_start_iter,
            pe_end_iter=pe_end_iter,
        )
        self.rgb_network = GSurfAppearanceNetwork(
            geo_feat_dim=geo_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=rgb_layers,
        )

    def set_iteration(self, iteration):
        self.sdf_network.set_iteration(iteration)

    def query_sdf_and_normal(self, points, create_graph=True):
        with torch.enable_grad():
            query_points = points.requires_grad_(True)
            sdf, geo_features = self.sdf_network(query_points)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=query_points,
                grad_outputs=torch.ones_like(sdf),
                create_graph=create_graph,
                retain_graph=True,
                only_inputs=True,
            )[0]
            normals = F.normalize(gradients, p=2, dim=-1, eps=1e-6)
        return sdf, normals, geo_features

    def query_color(self, points, camera_center, detach_points=True, create_graph=True):
        query_points = points.detach() if detach_points else points
        sdf, normals, geo_features = self.query_sdf_and_normal(
            query_points,
            create_graph=create_graph,
        )
        viewdirs = F.normalize(camera_center[None] - query_points, p=2, dim=-1, eps=1e-6)
        colors = self.rgb_network(query_points, viewdirs, normals, geo_features)
        return colors

    def save(self, model_path, iteration):
        save_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "iteration": iteration,
            },
            os.path.join(save_dir, "gsurf_networks.pth"),
        )

    def save_checkpoint_sidecar(self, checkpoint_path, iteration):
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "iteration": iteration,
            },
            checkpoint_path,
        )

    @classmethod
    def load(cls, path, device="cuda"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint.get("config", {})).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.set_iteration(checkpoint.get("iteration", model.config.get("pe_end_iter", 0)))
        return model

    @classmethod
    def load_from_model_path(cls, model_path, iteration, device="cuda"):
        path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "gsurf_networks.pth")
        if not os.path.exists(path):
            return None
        return cls.load(path, device=device)


@dataclass
class GSurfLossOutput:
    total: torch.Tensor
    position: torch.Tensor
    eikonal: torch.Tensor
    offsurface: torch.Tensor
    orientation: torch.Tensor
    normal_map: torch.Tensor
    entropy: torch.Tensor


def get_gaussian_normals(gaussians):
    rotations = build_rotation(gaussians.get_rotation)
    return F.normalize(torch.cross(rotations[:, :, 0], rotations[:, :, 1], dim=-1), p=2, dim=-1, eps=1e-6)


def opacity_entropy(opacity):
    opacity = opacity.clamp(1e-6, 1.0 - 1e-6)
    return (-opacity * torch.log(opacity) - (1.0 - opacity) * torch.log(1.0 - opacity)).mean()


def sample_uniform_in_bbox(points, count, padding_ratio=0.1):
    with torch.no_grad():
        pts = points.detach()
        bmin = pts.amin(dim=0)
        bmax = pts.amax(dim=0)
        extent = (bmax - bmin).clamp_min(1e-3)
        pad = extent.max() * padding_ratio
        bmin = bmin - pad
        bmax = bmax + pad
        samples = torch.rand(count, 3, device=points.device, dtype=points.dtype)
        samples = samples * (bmax - bmin)[None] + bmin[None]
    return samples


def sample_near_surface(points, count, scene_extent, noise_scale=0.01):
    with torch.no_grad():
        if points.shape[0] == 0:
            return points
        idx = torch.randint(0, points.shape[0], (count,), device=points.device)
        noise = torch.randn(count, 3, device=points.device, dtype=points.dtype)
        noise = noise * max(float(scene_extent) * noise_scale, 1e-4)
        return points.detach()[idx] + noise


def compute_gsurf_losses(
    gsurf_model,
    gaussians,
    opt,
    scene_extent,
    iteration,
    render_pkg=None,
    viewpoint_cam=None,
    render_func=None,
    pipe=None,
    background=None,
):
    device = gaussians.get_xyz.device
    zero = torch.zeros((), device=device)
    if iteration <= opt.gsurf_sdf_start_iter:
        return GSurfLossOutput(zero, zero, zero, zero, zero, zero, zero)

    surface_points = gaussians.get_xyz.detach() if opt.gsurf_detach_xyz else gaussians.get_xyz
    surface_points = surface_points.requires_grad_(True)
    sdf_values, sdf_normals, _ = gsurf_model.query_sdf_and_normal(surface_points, create_graph=True)

    position_loss = sdf_values.abs().mean()

    n_eik = max(int(opt.gsurf_num_eik_samples), 0)
    if n_eik > 0:
        uniform_count = n_eik // 2
        near_count = n_eik - uniform_count
        eik_points = []
        if uniform_count > 0:
            eik_points.append(sample_uniform_in_bbox(surface_points, uniform_count))
        if near_count > 0:
            eik_points.append(sample_near_surface(surface_points, near_count, scene_extent))
        eik_points = torch.cat(eik_points, dim=0).requires_grad_(True)
        eik_grad = gsurf_model.sdf_network.gradient(eik_points, create_graph=True)
        eikonal_loss = (torch.linalg.norm(eik_grad, dim=-1) - 1.0).pow(2).mean()
    else:
        eikonal_loss = zero

    n_off = max(int(opt.gsurf_num_off_samples), 0)
    if n_off > 0:
        off_points = sample_uniform_in_bbox(surface_points, n_off).requires_grad_(True)
        off_sdf = gsurf_model.sdf_network.sdf(off_points)
        offsurface_loss = torch.exp(-float(opt.gsurf_off_alpha) * off_sdf.abs()).mean()
    else:
        offsurface_loss = zero

    gaussian_normals = get_gaussian_normals(gaussians).detach()
    orientation_loss = 1.0 - torch.abs((gaussian_normals * sdf_normals).sum(dim=-1))
    orientation_loss = orientation_loss.mean()

    normal_map_loss = zero
    if (
        opt.gsurf_loss_normal > 0.0
        and render_pkg is not None
        and viewpoint_cam is not None
        and render_func is not None
        and pipe is not None
        and background is not None
    ):
        sdf_normal_pkg = render_func(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            sdf_grad=sdf_normals,
            detach_geometry=True,
        )
        sdf_normal_map = F.normalize(sdf_normal_pkg["rend_sdf_normal"], p=2, dim=0, eps=1e-6)
        target_normal_map = F.normalize(render_pkg["rend_normal"].detach(), p=2, dim=0, eps=1e-6)
        alpha = render_pkg["rend_alpha"].detach()
        normal_error = 1.0 - (sdf_normal_map * target_normal_map).sum(dim=0, keepdim=True).abs()
        normal_map_loss = (normal_error * alpha).sum() / alpha.sum().clamp_min(1.0)

    entropy_loss = opacity_entropy(gaussians.get_opacity)

    total = (
        opt.gsurf_loss_pos * position_loss
        + opt.gsurf_loss_eik * eikonal_loss
        + opt.gsurf_loss_off * offsurface_loss
        + opt.gsurf_loss_ori * orientation_loss
        + opt.gsurf_loss_normal * normal_map_loss
        + opt.gsurf_loss_entropy * entropy_loss
    )
    return GSurfLossOutput(
        total=total,
        position=position_loss,
        eikonal=eikonal_loss,
        offsurface=offsurface_loss,
        orientation=orientation_loss,
        normal_map=normal_map_loss,
        entropy=entropy_loss,
    )


@torch.no_grad()
def extract_gsurf_mesh(gsurf_model, gaussians, resolution=256, padding_ratio=0.05, level=0.0, chunk=64):
    from skimage import measure
    import trimesh

    points = gaussians.get_xyz.detach()
    bmin = points.amin(dim=0)
    bmax = points.amax(dim=0)
    extent = (bmax - bmin).clamp_min(1e-3)
    pad = extent.max() * padding_ratio
    bmin = bmin - pad
    bmax = bmax + pad

    xs = torch.linspace(bmin[0], bmax[0], resolution, device=points.device)
    ys = torch.linspace(bmin[1], bmax[1], resolution, device=points.device)
    zs = torch.linspace(bmin[2], bmax[2], resolution, device=points.device)
    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    x_chunks = xs.split(chunk)
    y_chunks = ys.split(chunk)
    z_chunks = zs.split(chunk)
    for xi, x_chunk in enumerate(x_chunks):
        x0 = xi * chunk
        for yi, y_chunk in enumerate(y_chunks):
            y0 = yi * chunk
            for zi, z_chunk in enumerate(z_chunks):
                z0 = zi * chunk
                xx, yy, zz = torch.meshgrid(x_chunk, y_chunk, z_chunk, indexing="ij")
                query = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
                sdf = gsurf_model.sdf_network.sdf(query).reshape(len(x_chunk), len(y_chunk), len(z_chunk))
                volume[
                    x0 : x0 + len(x_chunk),
                    y0 : y0 + len(y_chunk),
                    z0 : z0 + len(z_chunk),
                ] = sdf.detach().cpu().numpy()

    if volume.min() > level or volume.max() < level:
        raise ValueError(
            f"SDF level {level} is outside the evaluated range [{volume.min()}, {volume.max()}]."
        )

    spacing = ((bmax - bmin) / (resolution - 1)).detach().cpu().numpy()
    vertices, faces, normals, _ = measure.marching_cubes(volume, level=level, spacing=spacing)
    vertices = vertices + bmin.detach().cpu().numpy()[None]
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.render_utils import save_img_u8
from utils.gsurf_utils import GSurfModel, compute_gsurf_losses

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gsurf_model = None
    gsurf_optimizer = None
    if opt.gsurf:
        gsurf_model = GSurfModel(
            sphere_radius=opt.gsurf_sphere_radius,
            pe_freqs=opt.gsurf_sdf_pe_freqs,
            pe_start_iter=opt.gsurf_sdf_pe_start_iter,
            pe_end_iter=opt.gsurf_sdf_pe_end_iter,
        ).cuda()
        gsurf_optimizer = torch.optim.Adam(
            [
                {"params": gsurf_model.sdf_network.parameters(), "lr": opt.gsurf_sdf_lr},
                {"params": gsurf_model.rgb_network.parameters(), "lr": opt.gsurf_rgb_lr},
            ],
            eps=1e-15,
        )
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if gsurf_model is not None:
            gsurf_checkpoint = os.path.splitext(checkpoint)[0] + "_gsurf.pth"
            if os.path.exists(gsurf_checkpoint):
                print(f"Loading GSurf checkpoint {gsurf_checkpoint}")
                loaded_gsurf = GSurfModel.load(gsurf_checkpoint, device="cuda")
                gsurf_model.load_state_dict(loaded_gsurf.state_dict(), strict=True)
                gsurf_model.set_iteration(first_iter)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_gsurf_for_log = 0.0
    depth_path = os.path.join(dataset.model_path, 'depth')
    os.makedirs(depth_path, exist_ok=True)

    def render_with_gsurf(
        viewpoint,
        model,
        pipe_args,
        bg,
        scaling_modifier=1.0,
        override_color=None,
        sdf_grad=None,
        detach_geometry=False,
    ):
        if gsurf_model is not None and opt.gsurf_use_rgb_mlp:
            if override_color is None:
                create_graph = torch.is_grad_enabled()
                override_color = gsurf_model.query_color(
                    model.get_xyz,
                    viewpoint.camera_center,
                    detach_points=opt.gsurf_detach_xyz,
                    create_graph=create_graph,
                )
                if not create_graph:
                    override_color = override_color.detach()
        return render(
            viewpoint,
            model,
            pipe_args,
            bg,
            scaling_modifier,
            override_color=override_color,
            sdf_grad=sdf_grad,
            detach_geometry=detach_geometry,
        )

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if gsurf_model is not None:
            gsurf_model.set_iteration(iteration)

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if gsurf_optimizer is not None:
            gsurf_optimizer.zero_grad(set_to_none=True)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render_with_gsurf(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        gsurf_losses = None
        gsurf_loss = torch.zeros((), device="cuda")
        if gsurf_model is not None:
            gsurf_losses = compute_gsurf_losses(
                gsurf_model,
                gaussians,
                opt,
                scene.cameras_extent,
                iteration,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint_cam,
                render_func=render,
                pipe=pipe,
                background=background,
            )
            gsurf_loss = gsurf_losses.total

        # loss
        total_loss = loss + dist_loss + normal_loss + gsurf_loss
        
        total_loss.backward()

        iter_end.record()
        if iteration % 10000 == 0 or iteration in testing_iterations:
            normal_2dgs_detach = render_pkg['rend_normal'] * render_pkg['rend_alpha'].detach() # mutual affection
            save_img_u8(img=normal_2dgs_detach.permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5, pth=f'{depth_path}/surfel_normal_{iteration}.png')


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            if gsurf_losses is not None:
                ema_gsurf_for_log = 0.4 * gsurf_loss.item() + 0.6 * ema_gsurf_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "gsurf": f"{ema_gsurf_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/gsurf_loss', ema_gsurf_for_log, iteration)
                if gsurf_losses is not None:
                    tb_writer.add_scalar('gsurf/position', gsurf_losses.position.item(), iteration)
                    tb_writer.add_scalar('gsurf/eikonal', gsurf_losses.eikonal.item(), iteration)
                    tb_writer.add_scalar('gsurf/offsurface', gsurf_losses.offsurface.item(), iteration)
                    tb_writer.add_scalar('gsurf/orientation', gsurf_losses.orientation.item(), iteration)
                    tb_writer.add_scalar('gsurf/normal_map', gsurf_losses.normal_map.item(), iteration)
                    tb_writer.add_scalar('gsurf/opacity_entropy', gsurf_losses.entropy.item(), iteration)
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_with_gsurf, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if gsurf_model is not None:
                    print("[ITER {}] Saving GSurf networks".format(iteration))
                    gsurf_model.save(scene.model_path, iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if gsurf_optimizer is not None:
                    gsurf_optimizer.step()
                    gsurf_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if gsurf_model is not None:
                    gsurf_model.save_checkpoint_sidecar(
                        scene.model_path + "/chkpnt" + str(iteration) + "_gsurf.pth",
                        iteration,
                    )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training.
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")

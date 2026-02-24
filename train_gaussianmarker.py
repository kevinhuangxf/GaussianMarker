import os
import torch
import torch.nn as nn
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, modified_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import random
import copy
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from hidden.hidden_images import encoder, decoder, params, str2msg, msg2str, default_transform, NORMALIZE_IMAGENET, UNNORMALIZE_IMAGENET, EncoderWithJND, JND, encoder_with_jnd
from einops import reduce, repeat, rearrange
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.aug_utils import addNoise, Resize, compute_snr

import seaborn as sns
import matplotlib.pyplot as plt

import copy
from torchvision.transforms.v2 import JPEG
from models.pointnet_cls import get_model, get_loss
from models.pointnet_utils import feature_transform_reguliarzer, compute_snr, noise, rotate, translate, crop_out, noise_to_gaussians_color
from utils.system_utils import searchForMaxIteration
from collections import Counter

def create_message(input_msg, random_msg=False):
    # create message
    if random_msg:
        msg_ori = torch.randint(0, 2, (1, params.num_bits), device="cuda").bool() # b k
    else:
        msg_ori = torch.Tensor(str2msg(input_msg)).unsqueeze(0)
    msg_ori = msg_ori.cuda()
    # msg = 2 * msg_ori.type(torch.float) - 1 # b k
    return msg_ori

def uncertainty_estimation(gaussians, scene, pipe, background, uncertainty_use_gs_render=False):

    # uncertainty
    filter_out_grad = ["rotation", "opacity", "scale"]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    filter_out_idx = [name2idx[k] for k in filter_out_grad]

    viewpoint_cams = scene.getTrainCameras().copy()

    gaussians_params = gaussians.capture()[1:7]
    gaussians_params = [p for i, p in enumerate(gaussians_params) if i not in filter_out_idx]

    # off load to cpu to avoid oom with greedy algo
    # device = gaussians_params[0].device if num_views == 1 else "cpu"
    device = "cpu" # we have to load to cpu because of inflation

    # H_train = torch.zeros(sum(p.numel() for p in gaussians_params), device=gaussians_params[0].device, dtype=gaussians_params[0].dtype)
    H_train = torch.zeros(gaussians_params[0].shape[0], device=gaussians_params[0].device, dtype=gaussians_params[0].dtype)

    # Run heesian on training set
    for i, cam in enumerate(tqdm(viewpoint_cams, desc="Calculating diagonal Hessian on training views")):
        # if exit_func():
        #     raise RuntimeError("csm should exit early")

        if not uncertainty_use_gs_render:
            render_pkg = modified_render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            H_train += sum([reduce(p.grad.detach(), "n ... -> n", "sum") for p in gaussians_params])
        else:
            render_pkg = render(cam, gaussians, pipe, background)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            H_train += sum([reduce(torch.square(p.grad.detach()), "n ... -> n", "sum") for p in gaussians_params])

        gaussians.optimizer.zero_grad(set_to_none = True) 

    H_train = H_train.to(device)
    return H_train

def densify_gs_by_mask(gaussians, H_mask):
    selected_pts_mask = ~H_mask
    new_xyz = gaussians._xyz[selected_pts_mask]

    # random sample location
    N = 1
    stds = gaussians.get_scaling[selected_pts_mask].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3),device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(gaussians._rotation[selected_pts_mask]).repeat(N, 1, 1)
    new_xyz = gaussians.get_xyz[selected_pts_mask].repeat(N, 1) + torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

    new_features_dc = gaussians._features_dc[selected_pts_mask]
    new_features_rest = gaussians._features_rest[selected_pts_mask]
    new_opacities = gaussians._opacity[selected_pts_mask]
    new_scaling = gaussians._scaling[selected_pts_mask]
    new_rotation = gaussians._rotation[selected_pts_mask]
    gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    gaussians_origin_densified = copy.deepcopy(gaussians)

    return gaussians, gaussians_origin_densified

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    gaussians_origin = copy.deepcopy(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # init message
    msg_ori = create_message(args.input_msg)

    # get image size
    viewpoint_stack = scene.getTrainCameras()
    image_height = int(viewpoint_stack[0].image_height)
    image_width = int(viewpoint_stack[0].image_width)
    image_ratio = image_height / image_width

    # augmentation
    gaussian_noise = addNoise(0.1)
    resize = transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.9, 1.0), ratio=(image_ratio, image_ratio))
    jpeg = JPEG((50, 100))
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=0.1)
    aug_dict = {"gaussian_noise": gaussian_noise, "resize": resize, "jpeg": jpeg, "gaussian_blur": gaussian_blur}

    # uncertainty
    H_train = uncertainty_estimation(gaussians, scene, pipe, background, args.uncertainty_use_gs_render)
    H_threshold = torch.abs(H_train).mean() * args.H_threshold_factor # for low uncertainty
    H_mask = torch.abs(H_train) < H_threshold # torch.abs(H_train).mean() // 500 # for high uncertainty

    # densify points clone
    gaussians, gaussians_origin_densified = densify_gs_by_mask(gaussians, H_mask)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # add augmentation
        random_value = random.random()
        if random_value < 0.25:
            image = resize(image)
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # decode
        nom_image = NORMALIZE_IMAGENET(image.unsqueeze(0))
        ft = decoder(nom_image)

        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(ft, msg_ori)
        loss = 10 * loss + bce_loss

        decoded_msg = ft > 0 # b k -> b k
        accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k

        loss.backward()

        # freeze original 3dgs points
        origin_points = len(gaussians_origin._xyz)
        gaussians._xyz.grad[:origin_points] = 0
        gaussians._features_dc.grad[:origin_points] = 0
        gaussians._features_rest.grad[:origin_points] = 0
        gaussians._scaling.grad[:origin_points] = 0
        gaussians._rotation.grad[:origin_points] = 0
        gaussians._opacity.grad[:origin_points] = 0

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "xyz": f"{gaussians.get_xyz.shape}", "BCE Loss": f"{bce_loss:.{3}f}", "Bit Accuracy": f"{accs.sum().item() / params.num_bits:.{3}f}"})
                progress_bar.update(10)
                with torch.no_grad():
                    image_origin = render(viewpoint_cam, gaussians_origin, pipe, bg)["render"]

                # # Define the directory name
                # dir_name = os.path.join(scene.model_path, f"results_gaussianmarker")
                # # Check if the directory exists
                # if not os.path.exists(dir_name):
                #     # If not, create it
                #     os.makedirs(dir_name)

                # if args.save_vis:
                #     torchvision.utils.save_image(torch.cat((image_origin, image, (image-image_origin) * 10), dim=2), os.path.join(dir_name, '{0:05d}'.format(iteration) + ".png"))

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), msg_ori, aug_dict, args)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                scene_model_path = scene.model_path
                point_cloud_path = os.path.join(scene_model_path, "point_cloud_uncertainty_wm/iteration_{}".format(iteration))
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_wm.ply"))
                gaussians_origin.save_ply(os.path.join(point_cloud_path, "point_cloud_origin.ply"))
                gaussians_origin_densified.save_ply(os.path.join(point_cloud_path, "point_cloud_densified.ply"))
                gaussians.save_ply_partial(os.path.join(point_cloud_path, "point_cloud_perturb_only.ply"), origin_points)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, msg_ori, aug_dict, args):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        dir_name = os.path.join(scene.model_path, f"results_gaussianmarker")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                bit_acc_test = 0.0
                bit_acc_noise_test = 0.0
                bit_acc_resize_test = 0.0
                bit_acc_jpeg_test = 0.0
                bit_acc_blur_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # save test images
                    gt_name = os.path.join(dir_name, f"gt_{idx}.png")
                    # if not os.path.exists(gt_name):
                    #     torchvision.utils.save_image(gt_image, gt_name)
                    if args.save_vis:
                        torchvision.utils.save_image(torch.cat((image, (image-gt_image)*10), dim=1), 
                                                     os.path.join(dir_name, f"{iteration:05d}_{idx}.png"))

                    # test hidden
                    # test hidden with tensor image
                    image_wm = NORMALIZE_IMAGENET(image.unsqueeze(0))

                    ft = decoder(image_wm)
                    decoded_msg = ft > 0 # b k -> b k
                    accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k

                    print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
                    print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
                    print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")

                    bit_acc_test += accs.sum().item() / params.num_bits

                    # convert to PIL image
                    clip_img = torch.clamp(UNNORMALIZE_IMAGENET(image_wm), 0, 1)
                    clip_img = torch.round(255 * clip_img) / 255 
                    clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())

                    # degradation
                    for k, aug in aug_dict.items():
                        clip_img_aug = aug(clip_img)
                        ft = decoder(default_transform(clip_img_aug).unsqueeze(0).cuda())
                        decoded_msg_aug = ft > 0 # b k -> b k
                        accs_aug = (~torch.logical_xor(decoded_msg_aug, msg_ori)) # b k -> b k
                        print(f"Bit Accuracy Aug {k}: {accs_aug.sum().item() / params.num_bits}")

                        if k == "gaussian_noise":
                            bit_acc_noise_test += accs_aug.sum().item() / params.num_bits
                        elif k == "resize":
                            bit_acc_resize_test += accs_aug.sum().item() / params.num_bits
                        elif k == "jpeg":
                            bit_acc_jpeg_test += accs_aug.sum().item() / params.num_bits
                        elif k == "gaussian_blur":
                            bit_acc_blur_test += accs_aug.sum().item() / params.num_bits
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                bit_acc_test /= len(config['cameras'])
                bit_acc_noise_test /= len(config['cameras'])
                bit_acc_resize_test /= len(config['cameras'])
                bit_acc_jpeg_test /= len(config['cameras'])
                bit_acc_blur_test /= len(config['cameras'])
                print("Evaluation Bit Accuracy: {} noise {} resize {} jpeg {} blur {}".format(bit_acc_test, bit_acc_noise_test, bit_acc_resize_test, bit_acc_jpeg_test, bit_acc_blur_test))

        torch.cuda.empty_cache()

def training_3d_decoder(dataset, opt, pipe, testing_iterations, args):
    
    binary_message = create_message(args.input_msg)
    # binary_message = torch.randint(0, 2, (1, code_length)).float().cuda()
    code_length = binary_message.shape[1]
    zero_message = torch.zeros((1, code_length)).float().cuda()
    one_message = torch.ones((1, code_length)).float().cuda()

    pointnet_loss = get_loss()
    pointnet_adv = get_model(1, normal_channel=False, channel_num=14).cuda()
    pointnet_msg = get_model(code_length, normal_channel=False, channel_num=14).cuda()
    optimizer = torch.optim.Adam(
        # pointnet.parameters(),
        list(pointnet_adv.parameters()) + list(pointnet_msg.parameters()),
        lr=1e-4, # from 1e-5 to 1e-4
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999) # Added exponential scheduler

    gaussianmarker_model_path = os.path.join(args.model_path, "point_cloud_uncertainty_wm")
    loaded_iter = searchForMaxIteration(gaussianmarker_model_path)

    gaussians_origin = GaussianModel(dataset.sh_degree)
    gaussians_origin_path = os.path.join(gaussianmarker_model_path,
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud_densified.ply") # "point_cloud_origin.ply"
    gaussians_origin.load_ply(gaussians_origin_path)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians_wm_path = os.path.join(gaussianmarker_model_path,
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud_wm.ply") # "point_cloud_origin.ply"
    gaussians.load_ply(gaussians_wm_path)

    first_iter = 0
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # augmentation
    aug_list = [noise_to_gaussians_color]

    random_index = random.choice(range(len(aug_list)))
    gaussians_aug = aug_list[random_index](gaussians)

    xyz = torch.cat((gaussians_aug.get_xyz, gaussians_aug._features_dc.squeeze(1), gaussians_aug.get_opacity, gaussians_aug.get_scaling, gaussians_aug.get_rotation), dim=1)
    xyz_origin = torch.cat((gaussians_origin.get_xyz, gaussians_origin._features_dc.squeeze(1), gaussians_origin.get_opacity, gaussians_origin.get_scaling, gaussians_origin.get_rotation), dim=1)

    mean = 0
    std_dev = 1.0
    # noise_np = np.random.normal(mean, std_dev, xyz.shape//10)
    noise_np = np.random.normal(mean, std_dev, (xyz.shape[0]//10, 1))
    # Convert the NumPy array to a PyTorch tensor
    noise_torch = torch.tensor(noise_np, dtype=torch.float32).cuda()

    feat_noise = torch.zeros((noise_torch.shape[0], 13)).cuda()
    xyz_noise = torch.cat((noise_torch, feat_noise), dim=1)
    xyz = torch.cat((xyz, xyz_noise), dim=0)

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # random select points
        N = args.num_points
        xyz = xyz[torch.randperm(xyz.shape[0])[:N]]
        xyz2 = xyz_origin[torch.randperm(xyz_origin.shape[0])[:N]]
        xyz_origin = xyz_origin[torch.randperm(xyz_origin.shape[0])[:N]]
        
        # point-net
        xyz_ = torch.cat((xyz.unsqueeze(0), xyz_origin.unsqueeze(0)), dim=0).permute(0, 2, 1)
        xyz2_ = torch.cat((xyz2.unsqueeze(0), xyz_origin.unsqueeze(0)), dim=0).permute(0, 2, 1)
        pred, feat = pointnet_adv(xyz_)
        pred_, feat_ = pointnet_adv(xyz2_)
        pred2, feat2 = pointnet_msg(xyz_)
        if iteration % 10 == 0:
            aug_list = [noise, rotate, translate, crop_out]
            random_index = random.choice(range(len(aug_list)))
            gaussians_aug = aug_list[random_index](gaussians)
            xyz = torch.cat((gaussians_aug.get_xyz, gaussians_aug._features_dc.squeeze(1), gaussians_aug.get_opacity, gaussians_aug.get_scaling, gaussians_aug.get_rotation), dim=1)

        # adv losses
        bce_loss = F.binary_cross_entropy(pred, torch.cat((torch.ones((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0))
        bce_loss_ = F.binary_cross_entropy(pred_, torch.cat((torch.zeros((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0))
        
        # msg losses
        mean_pred2 = torch.mean(pred2, dim=0).unsqueeze(0)

        bce_loss2 = F.binary_cross_entropy(mean_pred2, binary_message)

        # feature regularization
        mat_diff_loss_scale = 0.001
        mat_diff_loss = feature_transform_reguliarzer(feat)
        mat_diff_loss2 = feature_transform_reguliarzer(feat2)

        lambda1_prime = 2.0

        # 1.0 * adv_loss + 2.0 * msg_loss
        loss = 1.0 * (bce_loss +  bce_loss_) + lambda1_prime * bce_loss2 + mat_diff_loss_scale * (mat_diff_loss  + mat_diff_loss2)

        loss.backward(retain_graph=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                # progress_bar.set_postfix({"pred": f"{(int(pred[0] > 0.5), int(pred[1] > 0.5))}"})
                # progress_bar.set_postfix({"binary_message": f"{binary_message}", "pred": f"{pred}"})
                adv_pred = (pred >= 0.5).float()
                adv_acc = torch.sum(adv_pred == torch.cat((torch.ones((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0)) / adv_pred.shape[0] / adv_pred.shape[1]
                adv_pred_ = (pred_ >= 0.5).float()
                adv_acc_ = torch.sum(adv_pred_ == torch.cat((torch.zeros((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0)) / adv_pred_.shape[0] / adv_pred_.shape[1]
                msg_pred = (pred2 >= 0.5).float()
                msg_acc = torch.sum(msg_pred == torch.cat((binary_message, binary_message), dim=0)) / msg_pred.shape[0] / msg_pred.shape[1]
                progress_bar.set_postfix({"loss": f"{loss}", "adv_acc": f"{adv_acc}", "adv_acc_": f"{adv_acc_}", "msg_acc": f"{msg_acc}"})
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                torch.cuda.empty_cache()

                def eval_aug(aug_index):
                    if aug_index == 4:
                        gaussians_aug = aug_list[aug_index](gaussians, 0.3)
                    else:
                        gaussians_aug = aug_list[aug_index](gaussians)

                    # point-net
                    xyz = torch.cat((gaussians_aug.get_xyz, gaussians_aug._features_dc.squeeze(1), gaussians_aug.get_opacity, gaussians_aug.get_scaling, gaussians_aug.get_rotation), dim=1)
                    xyz_origin = torch.cat((gaussians_origin.get_xyz, gaussians_origin._features_dc.squeeze(1), gaussians_origin.get_opacity, gaussians_origin.get_scaling, gaussians_origin.get_rotation), dim=1)
                            
                    # Select the corresponding points from xyz tensor
                    N = args.num_points
                    xyz = xyz[torch.randperm(xyz.shape[0])[:N]]
                    xyz2 = xyz_origin[torch.randperm(xyz.shape[0])[:N]]
                    xyz_origin = xyz_origin[torch.randperm(xyz_origin.shape[0])[:N]]

                    xyz_ = torch.cat((xyz.unsqueeze(0), xyz_origin.unsqueeze(0)), dim=0).permute(0, 2, 1)
                    xyz2_ = torch.cat((xyz_origin.unsqueeze(0), xyz2.unsqueeze(0)), dim=0).permute(0, 2, 1)
                    pred, feat = pointnet_adv(xyz_)
                    pred2, feat2 = pointnet_msg(xyz2_)
                    pred_, feat_ = pointnet_adv(xyz2_)
                    adv_pred = (pred >= 0.5).float()
                    adv_acc = torch.sum(adv_pred == torch.cat((torch.ones((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0)) / adv_pred.shape[0] / adv_pred.shape[1]
                    msg_pred = (pred2 >= 0.5).float()
                    msg_acc = torch.sum(msg_pred == torch.cat((binary_message, binary_message), dim=0)) / msg_pred.shape[0] / msg_pred.shape[1]
                    adv_pred_ = (pred_ >= 0.5).float()
                    adv_acc_ = torch.sum(adv_pred_ == torch.cat((torch.zeros((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0)) / adv_pred_.shape[0] / adv_pred_.shape[1]
                    print(f"Test iteration {iteration}: adv_acc: {adv_acc} adv_acc_: {adv_acc_} msg_acc: {msg_acc}")

                    return adv_acc, adv_acc_

                for aug_index in range(len(aug_list)):
                    print(aug_list[aug_index])
                    adv_pred_result = torch.zeros((2, 1)).float().cuda() # torch.cat((torch.ones((1, 1)).float().cuda(), torch.zeros((1, 1)).float().cuda()), dim=0) # torch.ones((2, 1)).float().cuda()
                    adv_pred__result = torch.zeros((2, 1)).float().cuda()

                    # majority voting
                    adv_acc_results = []
                    adv_acc__results = []
                    for i in range(11):
                        adv_acc, adv_acc_ = eval_aug(aug_index)
                        adv_acc_results.append(adv_acc)
                        adv_acc__results.append(adv_acc_)

                    counter = Counter(adv_acc_results)
                    most_common_item, count = counter.most_common(1)[0]
                    adv_acc = most_common_item

                    counter = Counter(adv_acc__results)
                    most_common_item, count = counter.most_common(1)[0]
                    adv_acc_ = most_common_item

                    print(f"Test iteration {iteration} results: adv_acc: {adv_acc} adv_acc_: {adv_acc_} msg_acc: {msg_acc}")

                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                optimizer.step()
                scheduler.step() # Exponential Scheduler
                optimizer.zero_grad()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000]) # 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--H_threshold_factor", type=int, default=0.002)
    parser.add_argument("--input_msg", type=str, default="111010110101000001010111010011010100010000100111")
    parser.add_argument("--save_vis", action='store_true', default=False)
    parser.add_argument("--uncertainty_use_gs_render", action='store_true', default=False)
    parser.add_argument("--train_3d_decoder", action='store_true', default=False)
    parser.add_argument("--num_points", type=int, default=10000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.train_3d_decoder:
        # dataset, opt, pipe, testing_iterations, args
        training_3d_decoder(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")

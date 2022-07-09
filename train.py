import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
import matplotlib.pyplot as plt
import cv2 as cv
import yaml
import math
import argparse
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

from models import NeRFModel, MLP_Spec, totalVariation, totalVariation_L2
from cfgnode import CfgNode
from dataloader import Data_Loader
from load_diligent import load_diligent
from load_lightstage import load_lightstage
from load_apple import load_apple
from position_encoder import get_embedding_function
from DirectionsToRusink import DirectionsToRusink
from train_depth_func import train_depth_func


def train(input_data, testing):
    batch_size = input_data['rgb'].size(0)
    input_xy = input_data['input_xy'][0].to(device)
    gt_normal = input_data['normal'][0].to(device)

    input_light_direction = input_data['input_light_direction'].view(-1, 3).to(device)
    gt_rgb = input_data['rgb'].view(-1, 1 if cfg.dataset.gray_scale else 3).to(device)
    gt_shadow_mask = input_data['shadow_mask'].view(-1, 1).to(device)
    if cfg.loss.contour_factor > 0:
        gt_contour = input_data['contour'][0].to(device)

    embed_input = encode_fn_input1(input_xy)
    if cfg.models.use_mean_var:
        mean_var = input_data['mean_var'][0].to(device)
        embed_input = torch.cat([embed_input, mean_var], dim=-1)

    output_normal_0, output_diff_0, output_spec_coeff_0 = model(embed_input)
    output_normal, output_diff, output_spec_coeff = \
        output_normal_0.repeat(batch_size, 1), output_diff_0.repeat(batch_size, 1), \
        output_spec_coeff_0.repeat(batch_size, 1)

    if cfg.models.use_specular:
        if cfg.models.specular.type == 'MLP_Spec':
            halfangle = DirectionsToRusink(light=input_light_direction, normal=output_normal, output_ch=cfg.models.specular.input_halfangle_ch)
            embed_halfangle = encode_fn_input3(halfangle)
            output_spe_basis = specular_model(embed_halfangle)
            output_spe = (torch.abs(output_spec_coeff)[..., None] * output_spe_basis).sum(dim=1)

        output_rho = output_diff + output_spe

    render_shading = F.relu((output_normal * input_light_direction).sum(dim=-1, keepdims=True))
    render_rgb = output_rho * render_shading

    if not testing:
        rgb_loss = rgb_loss_function(render_rgb * gt_shadow_mask, gt_rgb * gt_shadow_mask)
        loss = rgb_loss

        if epoch <= int(cfg.loss.regularize_epoches * end_epoch):  # if epoch is small, use tv to guide the network
            if cfg.loss.diff_tv_factor > 0:
                diff_color_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
                diff_color_map[idxp] = output_diff_0
                tv_loss = totalVariation(diff_color_map, mask, num_rays) * batch_size * cfg.loss.diff_tv_factor
                loss += tv_loss
            if cfg.loss.spec_tv_factor > 0:
                spec_color_map = torch.zeros((h, w, output_spec_coeff_0.size(1)), dtype=torch.float32, device=device)
                spec_color_map[idxp] = output_spec_coeff_0
                tv_loss = totalVariation(spec_color_map, mask, num_rays) * batch_size * cfg.loss.spec_tv_factor
                loss += tv_loss
            if cfg.loss.normal_tv_factor > 0:
                normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
                normal_map[idxp] = output_normal_0
                tv_loss = totalVariation_L2(normal_map, mask, num_rays) * batch_size * cfg.loss.normal_tv_factor
                loss += tv_loss
            if cfg.loss.contour_factor > 0:
                contour_normal_loss = torch.abs(output_normal_0[..., -1] * gt_contour).mean()
                loss += contour_normal_loss * cfg.loss.contour_factor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the running loss
        normal_loss = torch.arccos(torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)).mean()
        cost_t = time.time() - start_t
        est_time = cost_t / ((epoch - start_epoch) * iters_per_epoch + iter_num + 1) * (
                (end_epoch - epoch) * iters_per_epoch + iters_per_epoch - iter_num - 1)
        print(
            'epoch: %d,  iter: %2d/ %d, Training: %.4f, Val: %.4f  cost_time: %d m %2d s,  est_time: %d m %2d s' %
            (epoch, iter_num + 1, iters_per_epoch, loss.item(), normal_loss.item() / math.pi * 180, cost_t // 60, cost_t % 60,
             est_time // 60, est_time % 60))
        writer.add_scalar('Training loss', rgb_loss.item(), (epoch - 1) * iters_per_epoch + iter_num)
        writer.add_scalar('Val loss', normal_loss.item() / math.pi * 180, (epoch - 1) * iters_per_epoch + iter_num)
    else:
        rgb_loss = F.l1_loss(render_rgb.view(-1), gt_rgb.view(-1))
        normal_loss = torch.arccos(torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)).mean()
        print("Testing RGB L1: %.4f     Normal MAE: %.4f" % (
        rgb_loss.item() * 255., normal_loss.item() / math.pi * 180))

        if eval_idx == 1:
            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            temp_nor = output_normal_0.clone()
            temp_nor[..., 1:] = -temp_nor[..., 1:]
            normal_map[idxp] = (temp_nor + 1) / 2
            normal_map = normal_map.cpu().numpy()
            normal_map = (np.clip(normal_map * 255., 0, 255)).astype(np.uint8)[:, :, ::-1]
            cv.imwrite(os.path.join(log_path, 'est_normal.png'), normal_map)

            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            normal_map[idxp] = temp_nor
            normal_map = normal_map.cpu().numpy()
            np.save(os.path.join(log_path, 'est_normal.npy'), normal_map)

            normalerr_map = torch.zeros((h, w), dtype=torch.float32, device=device)
            normal_err = torch.arccos(
                torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)) / math.pi * 180
            normalerr_map[idxp] = torch.clamp(normal_err, max=50)
            normalerr_map = normalerr_map.cpu().numpy()
            plt.matshow(normalerr_map)
            plt.colorbar()
            plt.savefig(os.path.join(log_path, 'est_normal_err.png'), dpi=200)
            plt.close()

            rgb_map = torch.ones((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
            rgb_map[idxp] = output_diff_0 / output_diff_0.max()
            rgb_map = rgb_map.cpu().numpy()
            rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
            cv.imwrite(os.path.join(log_path, 'est_brdf_diff.png'), rgb_map)

            for i in range(cfg.models.specular.num_basis):
                rgb_map = torch.zeros((h, w), dtype=torch.float32, device=device)
                rgb_map[idxp] = output_spec_coeff_0[:, i]
                rgb_map = rgb_map.cpu().numpy()
                plt.matshow(rgb_map)
                plt.colorbar()
                plt.savefig(os.path.join(log_path, 'est_brdf_speccoeff%d.png' % i), dpi=200)
                plt.close()

            if output_spec_coeff.size(1) > cfg.models.specular.num_basis:
                s_idx = cfg.models.specular.num_basis
                i = 0
                while s_idx < output_spec_coeff.size(1):
                    rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
                    rgb_map[idxp] = torch.abs(output_spec_coeff_0[:, s_idx:s_idx+(1 if cfg.dataset.gray_scale else 3)])
                    rgb_map = rgb_map.cpu().numpy()
                    rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)
                    cv.imwrite(os.path.join(log_path, 'est_brdf_speccoeff%d_RGB.png' % i), rgb_map[:, :, ::-1])
                    s_idx = s_idx + (1 if cfg.dataset.gray_scale else 3)
                    i += 1

        if eval_idx >= 2:
            log_path_brdf_spec = os.path.join(log_path, 'brdf_spec')
            os.makedirs(log_path_brdf_spec, exist_ok=True)
            log_path_rgb = os.path.join(log_path, 'rgb')
            os.makedirs(log_path_rgb, exist_ok=True)
            log_path_brdf = os.path.join(log_path, 'brdf')
            os.makedirs(log_path_brdf, exist_ok=True)
            log_path_shading = os.path.join(log_path, 'shading')
            os.makedirs(log_path_shading, exist_ok=True)
        else:
            log_path_brdf_spec = log_path
            log_path_rgb = log_path
            log_path_brdf = log_path
            log_path_shading = log_path

        rgb_map = torch.ones((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[idxp] = output_spe[:len(idxp[0])] * render_shading[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_%03d.png' % eval_idx), rgb_map)

        ################ Get Specular on Unit Sphere ################
        if cfg.models.specular.type == 'MLP_Spec':
            unit_sphere_halfangle = DirectionsToRusink(
                light=input_light_direction[:1, :].repeat(unit_sphere_normal.size(0), 1),
                normal=unit_sphere_normal.to(device),
                output_ch=cfg.models.specular.input_halfangle_ch)
            embed_halfangle = encode_fn_input3(unit_sphere_halfangle)
            output_unit_spe_basis = specular_model(embed_halfangle)

        unit_shading = F.relu((unit_sphere_normal.to(device) * input_light_direction[:1, :].repeat(unit_sphere_normal.size(0), 1)).sum(dim=-1, keepdims=True))

        for basis_idx in range(output_unit_spe_basis.size(1)):
            rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
            rgb_map[unit_sphere_idxp] = output_unit_spe_basis[:, basis_idx, :] * unit_shading
            rgb_map = rgb_map / rgb_map.max()
            rgb_map = rgb_map.cpu().numpy()
            rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
            cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_basis%d_%03d.png' % (basis_idx, eval_idx)), rgb_map)
        rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[unit_sphere_idxp] = output_unit_spe_basis.sum(dim=1) * unit_shading
        rgb_map = rgb_map / rgb_map.max()
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_basisSum_%03d.png' % eval_idx), rgb_map)
        ########################################################

        rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)

        rgb_map[idxp] = render_rgb[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_rgb, 'est_rgb_%03d.png' % eval_idx), rgb_map)

        rgb_map = torch.zeros((h, w, 1), dtype=torch.float32, device=device)
        rgb_map[idxp] = render_shading[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_shading, 'est_shading_%03d.png' % eval_idx), rgb_map)

        render_shading[torch.where(render_shading > 0.001)] = 1
        rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[idxp] = (output_rho * render_shading)[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path_brdf, 'est_brdf_%03d.png' % eval_idx), rgb_map)

    return output_normal_0


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="configs/exp1/reading.yml",
        help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--testing", type=str2bool,
        default=False,
        help="Enable testing mode."
    )
    parser.add_argument(
        "--cuda", type=str,
        help="Cuda ID."
    )
    parser.add_argument(
        "--quick_testing", type=str2bool,
        default=False,
        help="Set it to False will have more visualization results. Set it to True will have less visualization results. "
    )
    configargs = parser.parse_args()

    if configargs.quick_testing:
        configargs.testing = True

    # Read config file.
    configargs.config = os.path.expanduser(configargs.config)
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if cfg.experiment.randomseed is not None:
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)
    if configargs.cuda is not None:
        cfg.experiment.cuda = "cuda:" + configargs.cuda
    else:
        cfg.experiment.cuda = "cuda:0"
    device = torch.device(cfg.experiment.cuda)

    log_path = os.path.expanduser(cfg.experiment.log_path)
    data_path = os.path.expanduser(cfg.dataset.data_path)

    if not configargs.testing:
        writer = SummaryWriter(log_path)
        copyfile(__file__, os.path.join(log_path, 'train.py'))
        copyfile(configargs.config, os.path.join(log_path, 'config.yml'))

    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))

    ##########################
    # Build data loader
    if 'pmsData' in data_path or 'DiLiGenT' in data_path:
        input_data_dict = load_diligent(data_path)
    elif 'LightStage' in data_path:
        input_data_dict = load_lightstage(data_path, scale=2)
    elif 'Apple' in data_path:
        input_data_dict = load_apple(data_path, scale=1)
    else:
        raise NotImplementedError('Unknown dataset')
    training_data_loader = Data_Loader(
        input_data_dict,
        gray_scale=cfg.dataset.gray_scale,
        data_len=input_data_dict['images'].shape[0],  #end_epoch - start_epoch + 1,
        shadow_threshold=cfg.dataset.shadow_threshold,
    )
    training_dataloader = torch.utils.data.DataLoader(training_data_loader, batch_size=batch_size, shuffle=not configargs.testing, num_workers=0)
    mask = training_data_loader.get_mask().to(device)
    mean_xy = training_data_loader.get_mean_xy().to(device)
    bounding_box_xy = training_data_loader.get_bounding_box()[0].to(device), training_data_loader.get_bounding_box()[1].to(device)
    unit_sphere_normal, unit_sphere_idxp = training_data_loader.get_unitsphere_normal()
    all_light_direction = training_data_loader.get_all_light_direction()
    images_max_value = training_data_loader.images_max.to(device)
    eval_data_len = len(training_data_loader) if configargs.testing else 1
    if configargs.quick_testing:
        eval_data_len = 1
        configargs.testing = True
    if cfg.experiment.eval_every_iter <= (end_epoch-start_epoch+1):
        eval_data_loader = Data_Loader(
            input_data_dict,
            gray_scale=cfg.dataset.gray_scale,
            data_len=eval_data_len,
            shadow_threshold=cfg.dataset.shadow_threshold,
        )
        eval_dataloader = torch.utils.data.DataLoader(eval_data_loader, batch_size=1, shuffle=False, num_workers=0)
    ##########################

    ##########################
    # Build model
    if cfg.models.use_mean_var:
        cfg.models.nerf.include_input_input2 += 2 if cfg.dataset.gray_scale else 6

    NeRFModel_output_ch = cfg.models.specular.num_basis
    model = NeRFModel(
        num_layers=cfg.models.nerf.num_layers,
        hidden_size=cfg.models.nerf.hidden_size,
        skip_connect_every=cfg.models.nerf.skip_connect_every,
        num_encoding_fn_input1=cfg.models.nerf.num_encoding_fn_input1,
        num_encoding_fn_input2=cfg.models.nerf.num_encoding_fn_input2,
        include_input_input1=cfg.models.nerf.include_input_input1,  # denote images coordinates (ix, iy)
        include_input_input2=cfg.models.nerf.include_input_input2,  # denote rgb latent code (lx, ly, lz)
        output_ch=NeRFModel_output_ch,
        gray_scale=cfg.dataset.gray_scale,
        mask=mask,
    )
    encode_fn_input1 = get_embedding_function(num_encoding_functions=cfg.models.nerf.num_encoding_fn_input1)
    model.train()
    model.to(device)

    if cfg.models.use_specular:
        if cfg.models.specular.type == 'MLP_Spec':
            specular_model = MLP_Spec(
                num_layers=cfg.models.specular.num_layers,
                hidden_size=cfg.models.specular.hidden_size,
                skip_connect_every=cfg.models.specular.skip_connect_every,
                num_encoding_fn_input=cfg.models.specular.num_encoding_fn_input,
                input_ch=cfg.models.specular.input_halfangle_ch,
                output_ch=cfg.models.specular.num_basis,
            )
            encode_fn_input3 = get_embedding_function(num_encoding_functions=cfg.models.specular.num_encoding_fn_input)

        specular_model.train()
        specular_model.to(device)

    params_list = []
    params_list.append({'params': model.parameters()})
    if cfg.models.use_specular:
        params_list.append({'params': specular_model.parameters()})
    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    ##########################

    ##########################
    # Load checkpoints
    if configargs.testing:
        cfg.models.load_checkpoint = True
        cfg.models.checkpoint_path = log_path
    if cfg.models.load_checkpoint:
        model_checkpoint_pth = os.path.expanduser(cfg.models.checkpoint_path)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found checkpoints', model_checkpoint_pth)
        ckpt = torch.load(model_checkpoint_pth, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if cfg.models.use_specular:
            specular_model.load_state_dict(ckpt['specular_model_state_dict'])
        start_epoch = ckpt['global_step']+1
    if configargs.testing:
        start_epoch = 1
        end_epoch = 1
        cfg.experiment.eval_every_iter = 1
        cfg.experiment.save_every_iter = 100
    ##########################

    if cfg.loss.rgb_loss == 'l1':
        rgb_loss_function = F.l1_loss
    elif cfg.loss.rgb_loss == 'l2':
        rgb_loss_function = F.mse_loss
    else:
        raise AttributeError('Undefined rgb loss function.')

    start_t = time.time()
    h, w = mask.size(0), mask.size(1)
    idxp = torch.where(mask > 0.5)
    num_rays = len(idxp[0])
    iters_per_epoch = len(training_dataloader)
    for epoch in range(start_epoch, end_epoch+1):
        for iter_num, input_data in enumerate(training_dataloader):
            if not configargs.testing:
                batch_size = int(eval(cfg.experiment.batch_size))
                output_normal_0 = train(input_data=input_data, testing=False)

        scheduler.step()

        if epoch == cfg.models.use_depth:
            print('************ Train Depth Network  *****************')
            train_depth_func(cfg, device, mask, all_light_direction, mean_xy, bounding_box_xy, input_data, target_normal=output_normal_0.detach())
            print('************ Using rendered shadow map ************')
            training_data_loader.update_valid_shadow_map_from_pth(path=os.path.join(log_path, 'depth/est_all_shadow_map.npy'), thres=cfg.dataset.shadow_threshold)

        if epoch % cfg.experiment.save_every_epoch == 0:
            savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'specular_model_state_dict': specular_model.state_dict() if cfg.models.use_specular else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, savepath)
            print('Saved checkpoints at', savepath)

        if epoch % cfg.experiment.eval_every_iter == 0:
            model.eval()
            with torch.no_grad():
                print('================ evaluation results===============')
                for eval_idx, eval_datain in enumerate(eval_dataloader, start=1):
                    batch_size = 1
                    train(input_data=eval_datain, testing=True)
                print('==================================================')
            model.train()



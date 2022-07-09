import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter
from models import MLP_base
from position_encoder import get_embedding_function
from cast_shadow import cast_shadow


def train_depth_func(cfg, device, mask, all_light_direction, mean_xy, bounding_box_xy, input_data, target_normal):
    log_path = os.path.join(os.path.expanduser(cfg.experiment.log_path), 'depth')
    writer = SummaryWriter(log_path)
    start_epoch, end_epoch = 1, 5000

    depth_model = MLP_base(
        num_layers=cfg.models.depth.num_layers,
        hidden_size=cfg.models.depth.hidden_size,
        skip_connect_every=cfg.models.depth.skip_connect_every,
        num_encoding_fn_input=cfg.models.depth.num_encoding_fn_input,
        input_ch=cfg.models.depth.include_input_input,
    )
    encode_fn_input_depth = get_embedding_function(num_encoding_functions=cfg.models.depth.num_encoding_fn_input)
    depth_model.train()
    depth_model.to(device)

    params_list = []
    params_list.append({'params': depth_model.parameters()})
    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)

    start_t = time.time()
    h, w = mask.size(0), mask.size(1)
    idxp = torch.where(mask > 0.5)
    input_xy = input_data['input_xy'][0].to(device)
    embed_input_patch = encode_fn_input_depth(input_xy)


    # Training depth model
    for epoch in range(start_epoch, end_epoch+1):
        px = input_data['px'][0].to(device)
        py = input_data['py'][0].to(device)

        output_depth_0 = depth_model(embed_input_patch)

        depth_map = torch.zeros((h, w), dtype=torch.float32, device=device)
        depth_map[idxp] = output_depth_0.view(-1)
        depth_map_dx = torch.zeros_like(depth_map)
        depth_map_dx[:,1:-1] = (depth_map[:, 2:] - depth_map[:, :-2]) * mask[:, 2:] * mask[:, :-2]
        depth_map_dy = torch.zeros_like(depth_map)
        depth_map_dy[1:-1,:] = (depth_map[2:, :] - depth_map[:-2, :]) * mask[2:, :] * mask[:-2, :]
        px[:, 2] = depth_map_dx[idxp]
        py[:, 2] = depth_map_dy[idxp]
        output_depth2normal_0 = -torch.cross(px, py, dim=-1)
        output_depth2normal_0 = F.normalize(output_depth2normal_0, p=2, dim=-1)

        depth2normal_loss = 1 - (output_depth2normal_0 * target_normal).sum(dim=-1).mean()
        loss = depth2normal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the running loss
        cost_t = time.time() - start_t
        est_time = cost_t / (epoch - start_epoch + 1) * (end_epoch - epoch - 1)
        if epoch % 100 == 0:
            print(
                'epoch: %5d/ %5d, Training: %.4f,  cost_time: %d m %2d s,  est_time: %d m %2d s' %
                (epoch, end_epoch, loss.item(), cost_t // 60, cost_t % 60, est_time // 60, est_time % 60))
            writer.add_scalar('Training loss', loss.item(), epoch - 1)

    # Save depth model
    savepath = os.path.join(log_path, 'depth_params_%05d.pth' % epoch)
    torch.save({
        'global_step': epoch,
        'depth_model_state_dict': depth_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, savepath)
    print('Saved checkpoints at', savepath)
    output_depth_0 = output_depth_0.detach()
    rgb_map = torch.zeros((h, w, 1), dtype=torch.float32, device=device)
    temp = output_depth_0.max() - output_depth_0
    rgb_map[idxp] = temp / temp.max()
    rgb_map = rgb_map.cpu().numpy()
    rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
    cv.imwrite(os.path.join(log_path, 'est_depth.png'), rgb_map)

    # Render shadows
    log_path_cast_shading = os.path.join(log_path, 'cast_shading')
    os.makedirs(log_path_cast_shading, exist_ok=True)

    out_render_shadow_map = []
    for light_id, light_direction in enumerate(all_light_direction):
        input_light_direction = light_direction[None, :].repeat(len(output_depth_0), 1).to(device)
        with torch.no_grad():
            img_xyz = torch.cat([input_xy, output_depth_0], dim=-1)
            render_cast_shadow = cast_shadow(
                img_xyz,
                input_light_direction,
                mask,
                sample_points=cfg.models.cast_shadow_sample_points,
                model=depth_model,
                encode_fn=encode_fn_input_depth,
                mean_xy=mean_xy,
                bounding_box_xy=bounding_box_xy
            )
            render_cast_shadow = render_cast_shadow.detach().cpu()
        out_render_shadow_map.append(render_cast_shadow.numpy())

        rgb_map = torch.zeros((h, w), dtype=torch.float32)
        rgb_map[idxp] = render_cast_shadow[:len(idxp[0])]
        rgb_map = rgb_map.numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :]
        cv.imwrite(os.path.join(log_path_cast_shading, 'est_cast_shading_%03d.png' % light_id), rgb_map)
    out_render_shadow_map = np.stack(out_render_shadow_map, axis=0)
    np.save(os.path.join(log_path, 'est_all_shadow_map.npy'), out_render_shadow_map)

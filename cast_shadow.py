import torch
import torch.nn.functional as F


def sample_points_along_light(img_xyz, light_xyz, num_points, bounding_box_xy):
    """
    sample points from image coordinates img_xyz , toward light_xyz direction, until hit the image boundry
    image boundry is defined as   x: 0~1 ,  y: 0~1
    Args:
        img_xyz: (batch, 3)
        light_xyz: (batch, 3)
        num_points:  int
    Returns:  (batch, num_points, 3)
    """
    bound_k = torch.zeros_like(img_xyz[:,0])

    if bounding_box_xy is not None:
        xmax, ymax = bounding_box_xy[0][0], bounding_box_xy[0][1]
        xmin, ymin = bounding_box_xy[1][0], bounding_box_xy[1][1]
    else:
        xmax, ymax = 1, 1
        xmin, ymin = 0, 0

    kx0 = (xmin - img_xyz[:,0]) / light_xyz[:,0]
    yx0 = img_xyz[:,1] + kx0 * light_xyz[:,1]
    bound_k = torch.where((kx0 > 0) & (yx0 > ymin) & (yx0 < ymax), kx0, bound_k)

    kx1 = (xmax - img_xyz[:,0]) / light_xyz[:,0]
    yx1 = img_xyz[:,1] + kx1 * light_xyz[:,1]
    bound_k = torch.where((kx1 > 0) & (yx1 > ymin) & (yx1 < ymax), kx1, bound_k)

    ky0 = (ymin - img_xyz[:,1]) / light_xyz[:,1]
    xy0 = img_xyz[:,0] + ky0 * light_xyz[:,0]
    bound_k = torch.where((ky0 > 0) & (xy0 > xmin) & (xy0 < xmax), ky0, bound_k)

    ky1 = (ymax - img_xyz[:,1]) / light_xyz[:,1]
    xy1 = img_xyz[:,0] + ky1 * light_xyz[:,0]
    bound_k = torch.where((ky1 > 0) & (xy1 > xmin) & (xy1 < xmax), ky1, bound_k)

    # k_steps = torch.linspace(start=0, end=1, steps=num_points+2, dtype=bound_k.dtype, device=bound_k.device)[1:-1]
    k_steps = torch.logspace(start=0, end=1, steps=num_points+2, base=2, dtype=bound_k.dtype, device=bound_k.device)[1:-1] - 1

    temp = (bound_k[...,None] * k_steps[None,...])  # (batch, 1)*(1, num_points)
    boundry = img_xyz[:,None,:] + temp[..., None] * light_xyz[:,None,:]
    #          (batch, 1, 3)      (batch, num_points, 1) (batch, 1, 3)
    return boundry


def mask_sample_ponts(mask, sample_points):
    """
    Mask out the sample_points, where its (x,y) coordinates is outside of the mask region
    Args:
        mask:  (H, W)   with 0 indicate out, and 1 indicate valid points
        sample_ponts: (batch, num_points, 3)
    Returns: (batch, num_points)  with 0 indicate out, and 1 indicate valid points
    """
    batch, num_points = sample_points.size(0), sample_points.size(1)
    sample_points_xy = sample_points.view(-1, 3)[:, :2].clone()
    sample_points_xy[:, 0] = (sample_points_xy[:, 0] - 0.5) * 2
    sample_points_xy[:, 1] = (sample_points_xy[:, 1] - 0.5) * 2   # convert to range -1~1
    sample_points_xy = sample_points_xy[None, None, ...]  # (1,1,batch*num_points,2)
    mask_value = F.grid_sample(mask[None, None, ...], sample_points_xy, mode='bilinear', padding_mode='border', align_corners=False)  # (1,1,1,batch*num_points)

    return mask_value.view(batch, num_points)


def cast_shadow_rendering(valid_points, model, encode_fn):
    """
    Input a series of points into the model, check its z value and model's output z values, output the cast-shadow value
    Args:
        valid_points:  (num_valid_pts, 3)
        model: model to call
        encode_fn: embedding function for x,y input
    Returns: (num_valid_pts,)     1 indicate not occluded,   0 indicate occluded,
    """
    input_xy = valid_points[:, :2]
    light_z = valid_points[:, -1:]

    embed_input = encode_fn(input_xy)
    pred_z = model(embed_input)

    out = torch.where(light_z < pred_z, torch.ones_like(light_z), torch.zeros_like(light_z))
    return out.view(-1)


def cast_shadow(img_xyz, light_xyz, mask, sample_points, model, encode_fn, mean_xy, bounding_box_xy):
    """
    Args:
        img_xyz: (batch, 3)
        light_xyz: (batch, 3)
        mask: (H, W)
        sample_points:  int
        model: model to call
        encode_fn: embedding function for x,y input
    Returns: (batch,)    0 indicate occluded,  1 indicate not occluded
    """
    with torch.no_grad():
        if sample_points == 0:
            return torch.ones_like(img_xyz[:, 0])

        img_xyz[:, :2] = img_xyz[:, :2] + mean_xy

        sampled_pts = sample_points_along_light(img_xyz, light_xyz, num_points=sample_points, bounding_box_xy=bounding_box_xy)  # (batch, num_points, 3)
        m = mask_sample_ponts(mask, sampled_pts)  # (batch, num_points)
        valid_pts = sampled_pts[torch.where(m > 0.9)]   # (valid_points, 3)

        valid_pts[:, :2] = valid_pts[:, :2] - mean_xy

        model_shadow_map = cast_shadow_rendering(valid_pts, model=model, encode_fn=encode_fn)  # (valid_points,)
        sampled_pts_shadow_mask = torch.ones_like(m)
        temp = torch.ones_like(sampled_pts_shadow_mask[torch.where(m > 0.9)])
        temp[torch.where(model_shadow_map < 0.9)] = 0
        sampled_pts_shadow_mask[torch.where(m > 0.9)] = temp   # (batch, num_points)
        out, _ = sampled_pts_shadow_mask.min(dim=-1)

    return out

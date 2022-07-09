import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import load_diligent


class Data_Loader(Dataset):
    def __init__(self, data_dict, gray_scale=False, data_len=1, use_rgbencoder=False, shadow_threshold=0.0, calibrated=True):
        self.light_intensity = torch.tensor(data_dict['light_intensity'], dtype=torch.float32)
        self.images = torch.tensor(data_dict['images'], dtype=torch.float32)  # (num_images, height, width, channel)
        self.calibrated = calibrated
        if self.calibrated:
            self.images = self.images / self.light_intensity[:, None, None, :]
        if gray_scale:
            self.images = self.images.mean(dim=-1, keepdim=True)  # (num_images, height, width, 1)
            self.light_intensity = self.light_intensity.mean(dim=-1, keepdim=True)
        self.images_max = torch.tensor(1.0, dtype=torch.float32)  # self.images.max()  #  torch.tensor(0.5, dtype=torch.float32)

        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        self.light_direction = torch.tensor(data_dict['light_direction'], dtype=torch.float32)

        self.gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)

        self.num_images = self.images.size(0)
        self.height = self.images.size(1)
        self.width = self.images.size(2)
        masks = self.mask[None,...].repeat((self.num_images,1,1))  # (num_images, height, width)

        self.valid_idx = torch.where(masks > 0.5)
        temp_idx = torch.where(self.mask > 0.5)

        self.valid_rgb = self.images[self.valid_idx]

        self.valid_input_iwih = torch.stack([temp_idx[1] / self.width, temp_idx[0] / self.height], dim=-1)
        self.valid_input_iwih_max, _ = self.valid_input_iwih.max(dim=0)
        self.valid_input_iwih_min, _ = self.valid_input_iwih.min(dim=0)
        self.mean_valid_iwih = self.valid_input_iwih.mean(dim=0, keepdim=True)

        self.valid_input_iwih = self.valid_input_iwih - self.mean_valid_iwih

        self.num_valid_rays = int(self.valid_input_iwih.size(0))
        self.valid_light_direction = torch.repeat_interleave(self.light_direction, self.num_valid_rays, dim=0)

        self.data_len = data_len
        self.use_rgbencoder = use_rgbencoder
        if self.use_rgbencoder:
            temp_imgs = self.images.permute(1,2,3,0)  # (height, width, channel, num_images)
            idx = torch.where(self.mask > 0.5)
            self.rgb_for_encoder = temp_imgs[idx]  # (num_valid_px_perimage, channel, num_images)
            # normalized rgb
            self.rgb_for_encoder = F.normalize(self.rgb_for_encoder, p=2, dim=-1)

        images_mean = torch.mean(self.images, dim=0)  # (height, width, channel)
        images_var = torch.var(self.images, dim=0)  # (height, width, channel)
        temp_mean_var = torch.cat([images_mean, images_var], dim=-1)  # (height, width, channel*2)
        self.valid_images_meanvar = temp_mean_var[temp_idx]

        self.valid_light_direction = self.valid_light_direction.view(self.num_images, -1, 3)
        self.valid_rgb = self.valid_rgb.view(self.num_images, -1, 1 if gray_scale else 3)
        self.valid_gt_normal = self.gt_normal[temp_idx]

        self.valid_shadow = None
        self.update_valid_shadow_map(thres=shadow_threshold)
        self.get_contour_idx()

    def __len__(self):
        return min(self.data_len, self.num_images)

    def __getitem__(self, idx):
        return self.get_testing_rays(idx)

    def get_testing_rays(self, ith):
        input_xy = self.valid_input_iwih
        input_light_direction = self.valid_light_direction[ith]
        rgb = self.valid_rgb[ith]
        normal = self.valid_gt_normal
        light_intensity = self.light_intensity[ith]

        mean_var = self.valid_images_meanvar

        sample = {'input_xy': input_xy,
                  'input_light_direction': input_light_direction,
                  'light_intensity': light_intensity,
                  'rgb': rgb,
                  'normal': normal,
                  'mean_var': mean_var}
        if self.use_rgbencoder:
            sample['rgb_for_encoder'] = self.rgb_for_encoder

        sample['shadow_mask'] = self.valid_shadow[ith]
        sample['contour'] = self.contour

        dx = 1 / self.mask.size(1)
        dy = 1 / self.mask.size(1)
        px = torch.zeros_like(input_light_direction)
        px[:, 0] = 2 * dx
        py = torch.zeros_like(input_light_direction)
        py[:, 1] = 2 * dy
        sample['px'] = px
        sample['py'] = py

        return sample

    def get_mask(self):
        return self.mask

    def get_mean_xy(self):
        return self.mean_valid_iwih

    def get_bounding_box(self):
        return self.valid_input_iwih_max, self.valid_input_iwih_min

    def get_all_light_direction(self):
        return self.light_direction

    def update_valid_shadow_map(self, thres):
        if self.valid_rgb.size(-1) == 3:
            temp_rgb = self.valid_rgb.mean(dim=-1)  # (num_image, num_mask_point)
        else:
            temp_rgb = self.valid_rgb
        temp_rgb_topk_mean = torch.topk(temp_rgb, k=len(temp_rgb)-11, dim=0, largest=False)[0].mean(dim=0, keepdim=True)

        idxp = torch.where(thres*temp_rgb_topk_mean <= temp_rgb)

        self.valid_shadow = torch.zeros_like(temp_rgb)
        self.valid_shadow[idxp] = 1
        return

    def update_valid_shadow_map_from_pth(self, path, thres):
        temp_render = torch.tensor(np.load(path), dtype=torch.float32)

        if self.valid_rgb.size(-1) == 3:
            temp_rgb = self.valid_rgb.mean(dim=-1)  # (num_image, num_mask_point)
        else:
            temp_rgb = self.valid_rgb
        temp_rgb_topk_mean = torch.topk(temp_rgb, k=len(temp_rgb)-11, dim=0, largest=False)[0].mean(dim=0, keepdim=True)

        idxp = torch.where(thres*temp_rgb_topk_mean <= temp_rgb)

        temp_thres = torch.zeros_like(temp_rgb)
        temp_thres[idxp] = 1

        self.valid_shadow = temp_thres * temp_render
        return

    def get_contour_idx(self):
        mask_x1, mask_x2, mask_y1, mask_y2 = self.mask.clone(), self.mask.clone(), self.mask.clone(), self.mask.clone()
        mask_x1[:-1, :] = self.mask[1:, :]
        mask_x2[1:, :] = self.mask[:-1, :]
        mask_y1[:, :-1] = self.mask[:, 1:]
        mask_y2[:, 1:] = self.mask[:, :-1]
        mask_1 = mask_x1 * mask_x2 * mask_y1 * mask_y2
        idxp_contour = torch.where((mask_1 < 0.5) & (self.mask > 0.5))

        contour_map = torch.zeros_like(self.mask)
        contour_map[idxp_contour] = 1

        self.contour = contour_map[torch.where(self.mask>0.5)]
        return idxp_contour


    @staticmethod
    def get_unitsphere_normal():
        data_dict = load_diligent.load_unitsphere()
        mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)
        valid_idx = torch.where(mask > 0.5)
        valid_gt_normal = gt_normal[valid_idx]
        return valid_gt_normal, valid_idx

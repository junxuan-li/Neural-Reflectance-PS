import cv2 as cv
import os
import numpy as np
import glob


def parse_txt(filename):
    out_list = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        for x in lines:
            lxyz = np.array([float(v) for v in x.strip().split()[1:]], dtype=np.float32)
            out_list.append(lxyz)
    out_arr = np.stack(out_list, axis=0).astype(np.float32)
    return out_arr


def load_lightstage(path, scale=1):
    if 'knight_standing' in path:  # or 'knight_fighting' in path:
        scale = 1
    images = []
    for img_file in sorted(glob.glob(os.path.join(path,"*[0-9]*.png"))):
        img = cv.imread(img_file)[:,:,::-1].astype(np.float32) / 255.
        if scale!=1:
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
        images.append(img)
    images = np.stack(images, axis=0)

    mask_files = glob.glob(os.path.join(path, "*_matte.png"))[0]
    mask_inv = cv.imread(mask_files, 0).astype(np.float32) / 255.
    mask = np.zeros_like(mask_inv)
    mask[np.where(mask_inv < 0.5)] = 1

    if scale!=1:
        mask = cv.resize(mask, dim, interpolation=cv.INTER_LINEAR)

    light_dir_files = os.path.join(os.path.dirname(path), "light_directions.txt")
    light_dir = parse_txt(light_dir_files)
    light_dir[..., 0] = -light_dir[..., 0]  # convert x-> -x

    light_intensity_files = os.path.join(os.path.dirname(path), "light_intensities.txt")
    light_intensity = parse_txt(light_intensity_files)

    gt_normal = np.zeros_like(images[0])

    idx = np.where(light_dir[..., -1] < -0.1)
    light_dir = light_dir[idx]
    light_intensity = light_intensity[idx]
    images = images[idx]
    print('Only use front size lights ld_z<-0.1, total images: %d' % len(light_dir))

    out_dict = {'images': images, 'mask': mask, 'light_direction': light_dir, 'light_intensity': light_intensity, 'gt_normal': gt_normal}
    return out_dict


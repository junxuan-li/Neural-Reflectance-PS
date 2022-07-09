import torch
import torch.nn as nn
import torch.nn.functional as F


# Model
class MLP_base(nn.Module):
    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=[4],
            num_encoding_fn_input=4,
            input_ch=3,
            output_ch=1,
    ):
        super(MLP_base, self).__init__()
        self.D = num_layers
        self.W = hidden_size
        self.input_ch = input_ch * (1 + 2 * num_encoding_fn_input)
        if type(skip_connect_every) != 'list':
            skip_connect_every = [skip_connect_every]
        self.skips = skip_connect_every
        # self.relu = torch.nn.functional.leaky_relu
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, hidden_size)] + [nn.Linear(hidden_size, hidden_size) if i not in self.skips else nn.Linear(hidden_size + self.input_ch, hidden_size) for i in
                                        range(num_layers - 1)])

        self.output_ch = output_ch
        self.output_linear = nn.Linear(hidden_size, self.output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = self.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        # outputs = torch.sigmoid(self.output_linear(h)) * 2
        outputs = -torch.abs(self.output_linear(h))
        return outputs


class MLP_Spec(nn.Module):
    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=[4],
            num_encoding_fn_input=4,
            input_ch=3,
            output_ch=1,
            gray_scale=False,
    ):
        super(MLP_Spec, self).__init__()
        self.D = num_layers
        self.W = hidden_size
        self.input_ch = input_ch * (1 + 2 * num_encoding_fn_input)
        if type(skip_connect_every) != 'list':
            skip_connect_every = [skip_connect_every]
        self.skips = skip_connect_every
        # self.relu = torch.nn.functional.leaky_relu
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, hidden_size)] + [nn.Linear(hidden_size, hidden_size) if i not in self.skips else nn.Linear(hidden_size + self.input_ch, hidden_size) for i in
                                        range(num_layers - 1)])

        self.output_ch = output_ch
        self.output_linear = nn.Linear(hidden_size, self.output_ch * (1 if gray_scale else 3))

    def forward(self, x):
        h = x
        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = self.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        outputs = torch.abs(self.output_linear(h))   # tanh range(-1,1)

        outputs = outputs.view(outputs.size(0), self.output_ch, -1)   # (batch, num_basis, channel)
        return outputs


class NeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        num_encoding_fn_input1=10,
        num_encoding_fn_input2=0,
        include_input_input1=2,   # denote images coordinates (ix, iy)
        include_input_input2=0,
        output_ch=1,
        gray_scale=False,
        mask=None,
    ):
        super(NeRFModel, self).__init__()
        self.dim_ldir = include_input_input2 * (1 + 2 * num_encoding_fn_input2)
        self.dim_ixiy = include_input_input1 * (1 + 2 * num_encoding_fn_input1) + self.dim_ldir
        self.dim_ldir = 0
        self.skip_connect_every = skip_connect_every + 1

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        # self.relu = torch.nn.functional.leaky_relu
        self.relu = torch.nn.functional.relu
        self.mask = mask
        self.idxp = torch.where(self.mask > 0.5)

        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_ldir, hidden_size // 2))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        self.fc_spec_coeff = torch.nn.Linear(hidden_size//2,  output_ch)

        self.fc_diff = torch.nn.Linear(hidden_size // 2, 1 if gray_scale else 3)

        self.fc_normal_xy = torch.nn.Linear(hidden_size, 2)
        self.fc_normal_z = torch.nn.Linear(hidden_size, 1)

    def get_est_light(self):
        num_rays = len(self.idxp[0])
        out_ld = torch.cat([self.light_direction_xy, -torch.abs(self.light_direction_z)], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)[:, None, :]    # (96, 1, 3)

        out_ld = out_ld.repeat(1, num_rays, 1)
        out_ld = out_ld.view(-1, 3)  # (96*num_rays, 3)

        out_li = torch.abs(self.light_intensity*10)[:, None, :]    # (96, 1, 3)
        out_li = out_li.repeat(1, num_rays, 1)
        out_li = out_li.view(-1, 3)  # (96*num_rays, 3)
        return out_ld, out_li

    def forward(self, input):
        xyz = input[..., : self.dim_ixiy]
        x = xyz
        for i in range(len(self.layers_xyz)):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)

        normal_xy = self.fc_normal_xy(x)
        normal_z = -torch.abs(self.fc_normal_z(x))  # n_z is always facing camera
        normal = torch.cat([normal_xy, normal_z], dim=-1)
        normal = F.normalize(normal, p=2, dim=-1)

        feat = self.fc_feat(x)
        if self.dim_ldir > 0:
            light_xyz = input[..., -self.dim_ldir:]
            feat = torch.cat([feat, light_xyz], dim=-1)
        x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_dir)):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        spec_coeff = self.fc_spec_coeff(x)

        diff = torch.abs(self.fc_diff(x))

        return normal, diff, spec_coeff


def totalVariation(image, mask, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var


def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var

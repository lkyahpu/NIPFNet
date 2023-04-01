import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
from function import DenseBlock_light, ConvLayer, Fusion_network, Dense_encoder, FastGuidedFilter_attention,FusionBlock_res
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate, softmax
import math
from fusion import AFF, iAFF, DAF,MS_CAM
from models import common
from imageio import imread, imsave


class pd_log(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(pd_log, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.dropout = nn.Dropout2d(p=0.5)
        # self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = F.sigmoid(out)

        return out


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class c_Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(x, y, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class c_PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, y):
        if self.shifted:
            x = self.cyclic_shift(x)
            y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv2 = self.to_qkv(y).chunk(3, dim=-1)

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, _, _ = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        _, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv2)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0].type(torch.long), self.relative_indices[:, :, 1].type(torch.long)]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class Cross_SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = c_Residual(c_PreNorm(dim, Cross_WindowAttention(dim=dim,
                                                                               heads=heads,
                                                                               head_dim=head_dim,
                                                                               shifted=shifted,
                                                                               window_size=window_size,
                                                                               relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y):
        x = self.attention_block(x, y)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, dim_scale=2):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(in_channels, out_channels * (dim_scale ** 2),
                                bias=False)  # if dim_scale==2 else nn.Identity()
        self.dim = out_channels
        # self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'B H W (p1 p2 c)-> B (H p1) (W p2) c', p1=self.dim_scale, p2=self.dim_scale, c=self.dim)
        x = x.view(B, H * self.dim_scale, W * self.dim_scale, self.dim)

        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers=2, downscaling_factor=1, num_heads=3, head_dim=32,
                 window_size=7,
                 relative_pos_embedding=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class Up_StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers=2, upscaling_factor=1, num_heads=3, head_dim=32,
                 window_size=7,
                 relative_pos_embedding=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=in_channels,
                                            downscaling_factor=1)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=in_channels, heads=num_heads, head_dim=head_dim, mlp_dim=in_channels * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=in_channels, heads=num_heads, head_dim=head_dim, mlp_dim=in_channels * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

        self.patch_expand = PatchExpand(in_channels=in_channels, out_channels=hidden_dimension,
                                        dim_scale=upscaling_factor)

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.patch_expand(x)

        return x.permute(0, 3, 1, 2)


class Cross_FusionModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers=2, downscaling_factor=1, num_heads=3, head_dim=32,
                 window_size=7,
                 relative_pos_embedding=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition1 = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                             downscaling_factor=downscaling_factor)
        self.patch_partition2 = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                             downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                Cross_SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                                shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                Cross_SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                                shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

        self.layers2 = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers2.append(nn.ModuleList([
                Cross_SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                                shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                Cross_SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                                shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x, y):

        x = self.patch_partition1(x)
        y = self.patch_partition2(y)

        for regular_block, shifted_block in self.layers:
            x1 = regular_block(x, y)
            # x = shifted_block(x)

        for regular_block2, shifted_block2 in self.layers2:
            y1 = regular_block2(y, x)
            # x = shifted_block2(x)

        x2 = shifted_block(x1, y1)
        y2 = shifted_block2(y1, x1)

        # for regular_block, shifted_block in self.layers:
        #     x1 = regular_block(x,y)
        #     y1 = regular_block(y,x)
        #     x2 = shifted_block(x1, y1)
        #     y2 = shifted_block(y1, x1)

        # x1 = self.layers[0](x,y)
        # y1 = self.layers2[0](y,x)
        #
        # x2 = self.layers[1](x1, y1)
        # y2 = self.layers2[1](y1, x1)

        return x2.permute(0, 3, 1, 2), y2.permute(0, 3, 1, 2)


class Cross_FusionBlock(torch.nn.Module):
    def __init__(self, channels, hidden_dim, layers=2, downscaling_factor=1, num_heads=3, head_dim=32,
                 window_size=7,
                 relative_pos_embedding=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        ws = [3, 3, 3, 3]

        kernel_size = 3
        self.num = kernel_size ** 2

        scale = 2

        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)

        self.SwinB_x = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers,
                                   downscaling_factor=downscaling_factor, num_heads=num_heads, head_dim=head_dim,
                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.SwinB_y = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers,
                                   downscaling_factor=downscaling_factor, num_heads=num_heads, head_dim=head_dim,
                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.SwinB_x = ConvLayer(hidden_dim, hidden_dim, 1, 1)
        # self.SwinB_y = ConvLayer(hidden_dim, hidden_dim, 1, 1)

        self.CFM = Cross_FusionModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers,
                                      downscaling_factor=downscaling_factor, num_heads=num_heads, head_dim=head_dim,
                                      window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.CFM = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers,
        #                            downscaling_factor=downscaling_factor, num_heads=num_heads, head_dim=head_dim,
        #                            window_size=window_size, relative_pos_embedding=relative_pos_embedding)


        # self.conv_1x1 = ConvLayer(hidden_dim * 2, 1, 1, 1)
        self.conv_1x1 = DenseBlock_light(hidden_dim, 1, 3, 1)

        self.convx_1x1 = common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=1, kernel_size=3,
                            padding=1, norm='Adaptive')

        self.convy_1x1 = common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=1, kernel_size=3,
                                             padding=1, norm='Adaptive')

        self.depth_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
            common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=self.num, kernel_size=1)
        )

        self.guide_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
            common.ConvBNReLU2D(in_channels=hidden_dim, out_channels=self.num, kernel_size=1)
        )



        self.conv_1x1_x = ConvLayer(hidden_dim, self.num, 1, 1)
        self.conv_1x1_y = ConvLayer(hidden_dim, self.num, 1, 1)
        self.conv_fusion = ConvLayer(hidden_dim * 2, hidden_dim, 1, 1)

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))
        self.conv_fusion = ConvLayer(2*channels, channels, 1, 1)
        #
        # self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        # self.conv_vi = ConvLayer(channels, channels, ws[index], 1)
        #
        # block = []
        # block += [ConvLayer(2*channels, channels, 1, 1),
        #           ConvLayer(channels, channels, ws[index], 1),
        #           ConvLayer(channels, channels, ws[index], 1)]
        # self.bottelblock = nn.Sequential(*block)
        # self.GRF = Sobelxy(2*channels)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        b, c, h, w = x_ir.size()

        c_x, c_y = self.CFM(x_ir, x_vi)

        # weight_map = self.conv_1x1(torch.cat((c_x, c_y), 1))  # wu Softmax

        # weight_map1 = self.convx_1x1(torch.cat((c_x, c_y), 1))
        weight_map1 = self.convx_1x1(c_x)
        weight_map2 = self.convy_1x1(c_y)


        x_kernel = self.SwinB_x(x_ir)
        y_kernel = self.SwinB_y(x_vi)

        # x_kernel = self.conv_1x1_x(x_kernel)
        # y_kernel = self.conv_1x1_y(y_kernel)

        x_kernel = self.depth_kernel(x_kernel)
        y_kernel = self.guide_kernel(y_kernel)


        x_kernel = softmax(x_kernel, dim=1)
        y_kernel = softmax(y_kernel, dim=1)

        # fuse_kernel = weight_map1 * x_kernel + (1 - weight_map1) * y_kernel

        fuse_kernel = weight_map1 * x_kernel + weight_map2 * y_kernel

        fuse_kernel = torch.tanh(fuse_kernel) / (self.aff_scale_const + 1e-8)

        abs_kernel = torch.abs(fuse_kernel)

        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4

        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        fuse_kernel = fuse_kernel / abs_kernel_sum

        # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
        unfold_inputs = self.unfold(self.conv_fusion(torch.cat((x_ir, x_vi), 1))).view(b, c, -1, h, w)
        # unfold_inputs = self.unfold(x_vi).view(b, c, -1, h, w)





        # # weight_map2 = 1- weight_map1
        # weight_map1 = weight_map1.detach().cpu().numpy().squeeze()
        # weight_map1 = (weight_map1 - np.min(weight_map1)) / (np.max(weight_map1) - np.min(weight_map1))
        # # weight_map1 = weight_map1.astype('uint8')
        #
        #
        # weight_map2 = weight_map2.detach().cpu().numpy().squeeze()
        # weight_map2 = (weight_map2 - np.min(weight_map2)) / (np.max(weight_map2) - np.min(weight_map2))
        #
        # w_path = 'test_weight'
        #
        # plt.figure('weight_map1')
        # plt.imshow(weight_map1,cmap='bwr')
        # plt.axis('off')
        # plt.colorbar()
        # plt.savefig(w_path+'/A_map.png')
        #
        # plt.figure('weight_map2')
        # plt.imshow(weight_map2,cmap='bwr')
        # plt.axis('off')
        # plt.colorbar()
        # plt.savefig(w_path + '/1-A_map.png')



        out = torch.einsum('bkhw, bckhw->bchw', [fuse_kernel, unfold_inputs])

        return out


class Cross_SwinT(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.Fusion1 = Cross_FusionBlock(channels=hidden_dim, hidden_dim=hidden_dim)
        self.Fusion2 = Cross_FusionBlock(channels=hidden_dim * 2, hidden_dim=hidden_dim * 2)
        self.Fusion3 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)
        self.Fusion4 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)

    def forward(self, en_x, en_y):
        en_x1 = self.Fusion1(en_x[0], en_y[0])
        en_x2 = self.Fusion2(en_x[1], en_y[1])
        en_x3 = self.Fusion3(en_x[2], en_y[2])
        en_x4 = self.Fusion4(en_x[3], en_y[3])

        en = [en_x1, en_x2, en_x3, en_x4]

        return en


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=1, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(1, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim * 8),
        #     nn.Linear(hidden_dim * 8, num_classes)
        # )
        # self.dense_encoder=Dense_encoder()

    def forward(self, img):
        en_x1 = self.stage1(img)
        en_x2 = self.stage2(en_x1)
        en_x3 = self.stage3(en_x2)
        en_x4 = self.stage4(en_x3)

        en = [en_x1, en_x2, en_x3, en_x4]

        return en


class Unet_SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=1, num_classes=1000, head_dim=32, window_size=7,
                 scaling_factors=(1, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=1, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=scaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=scaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=scaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
                                  downscaling_factor=scaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.stage4_4 = Up_StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
        #                                upscaling_factor=scaling_factors[3], num_heads=heads[3], head_dim=head_dim,
        #                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        #
        # self.stage3_3 = Up_StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 2, layers=layers[2],
        #                                upscaling_factor=scaling_factors[2], num_heads=heads[2], head_dim=head_dim,
        #                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        #
        # self.stage2_2 = Up_StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim, layers=layers[1],
        #                                upscaling_factor=scaling_factors[1], num_heads=heads[1], head_dim=head_dim,
        #                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        #
        # self.stage1_1 = Up_StageModule(in_channels=hidden_dim * 1, hidden_dimension=hidden_dim, layers=layers[0],
        #                                upscaling_factor=scaling_factors[0], num_heads=heads[0], head_dim=head_dim,
        #                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        #
        # self.last_1x1 = ConvLayer(hidden_dim, 1, 1, 1)

        # self.conv_out3 = ConvLayer(hidden_dim * 8, hidden_dim * 4, 1, 1)
        # self.conv_out2 = ConvLayer(hidden_dim * 4, hidden_dim * 2, 1, 1)
        # self.conv_out1 = ConvLayer(hidden_dim * 2, hidden_dim * 1, 1, 1)

        self.Guide1 = Cross_FusionBlock(channels=hidden_dim, hidden_dim=hidden_dim)
        self.Guide2 = Cross_FusionBlock(channels=hidden_dim * 2, hidden_dim=hidden_dim * 2)
        self.Guide3 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)
        self.Guide4 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)

        self.iAFF1 = iAFF(hidden_dim)
        self.iAFF2 = iAFF(hidden_dim * 2)
        self.iAFF3 = iAFF(hidden_dim * 4)
        self.iAFF4 = iAFF(hidden_dim * 4)

        # self.fusion_block1 = MS_CAM(hidden_dim)
        # self.fusion_block2 = MS_CAM(hidden_dim * 2)
        # self.fusion_block3 = MS_CAM(hidden_dim * 4)
        # self.fusion_block4 = MS_CAM(hidden_dim * 4)



        # self.Guide1_1 = Cross_FusionBlock(channels=hidden_dim, hidden_dim=hidden_dim)
        # self.Guide2_2 = Cross_FusionBlock(channels=hidden_dim * 2, hidden_dim=hidden_dim * 2)
        # self.Guide3_3 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)
        #
        # self.gf = FastGuidedFilter_attention(r=17, eps=1e-5)
        #
        # self.conv_out = ConvLayer(hidden_dim, 1, 1, 1, is_last=True)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim * 8),
        #     nn.Linear(hidden_dim * 8, num_classes)
        # )
        # self.dense_encoder=Dense_encoder()

    def forward(self, rho, pd, I):


        # en_x1_p = self.stage1(torch.cat([rho,pd],1))
        en_x1_pd  = self.stage1(pd)
        en_x1_rho = self.stage1(rho)
        en_x1 = self.iAFF1(en_x1_pd,en_x1_rho)
        # en_x1 = self.fusion_block1(en_x1_dop)
        en_x1 = self.Guide1(I[0], en_x1)


        # en_x2_p = self.stage2(en_x1_p)
        en_x2_pd  = self.stage2(en_x1_pd)
        en_x2_rho = self.stage2(en_x1_rho)
        en_x2 = self.iAFF2(en_x2_pd,en_x2_rho)
        # en_x2 = self.fusion_block2(en_x2_dop)
        en_x2 = self.Guide2(I[1], en_x2)


        # en_x3_p = self.stage3(en_x2_p)
        en_x3_pd = self.stage3(en_x2_pd)
        en_x3_rho = self.stage3(en_x2_rho)
        en_x3 = self.iAFF3(en_x3_pd,en_x3_rho)
        # en_x3 = self.fusion_block3(en_x3_dop)
        en_x3 = self.Guide3(I[2], en_x3)


        # en_x4_p = self.stage4(en_x3_p)
        en_x4_pd = self.stage4(en_x3_pd)
        en_x4_rho = self.stage4(en_x3_rho)
        en_x4 = self.iAFF4(en_x4_pd,en_x4_rho)
        # en_x4 = self.fusion_block4(en_x4_dop)
        en_x4 = self.Guide4(I[3], en_x4)



        # en_x1 = self.stage1(torch.cat([img,pd],1))
        # en_x1 = self.Guide1(I[0], en_x1)
        #
        # en_x2 = self.stage2(en_x1)
        # en_x2 = self.Guide2(I[1], en_x2)
        #
        # en_x3 = self.stage3(en_x2)
        # en_x3 = self.Guide3(I[2], en_x3)
        #
        # en_x4 = self.stage4(en_x3)
        # en_x4 = self.Guide4(I[3], en_x4)




        # c_x, c_y = self.gf_att_map(x, pd)
        #
        # att_map = self.gf_1x1(torch.cat((c_x, c_y), 1))
        #
        # # att_map = F.relu(theta_x + phi_g, inplace=True)
        #
        # att_map = F.sigmoid(att_map)
        #
        # y_gf = self.gf(x, y, att_map)

        # dop_ir = torch.cat([dop,ir],1)

        # en_x4_4 = self.stage4_4(en_x4)
        # en_x3_3 = self.stage3_3(self.Guide3_3(I[2], en_x4_4))
        # en_x2_2 = self.stage2_2(self.Guide2_2(I[1], en_x3_3))
        # en_x1_1 = self.stage1_1(self.Guide1_1(I[0], en_x2_2))
        #
        # en_x1_1 = self.conv_out(en_x1_1)

        en = [en_x1, en_x2, en_x3, en_x4]

        # img_denoise = self.last_1x1(en_x1_1)

        return en




class UP_SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim=32, layers=(2, 2, 2, 2), heads=(3, 3, 3, 3), channels=1, num_classes=1000, head_dim=32, window_size=7,
                 scaling_factors=(1, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage4_4 = Up_StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
                                       upscaling_factor=scaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage3_3 = Up_StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 2, layers=layers[2],
                                       upscaling_factor=scaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage2_2 = Up_StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim, layers=layers[1],
                                       upscaling_factor=scaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.stage1_1 = Up_StageModule(in_channels=hidden_dim * 1, hidden_dimension=hidden_dim, layers=layers[0],
                                       upscaling_factor=scaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                       window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.last_1x1 = ConvLayer(hidden_dim, 1, 1, 1)

        self.conv_out3 = ConvLayer(hidden_dim * 8, hidden_dim * 4, 1, 1)
        self.conv_out2 = ConvLayer(hidden_dim * 4, hidden_dim * 2, 1, 1)
        self.conv_out1 = ConvLayer(hidden_dim * 2, hidden_dim * 1, 1, 1)


        self.conv_out = ConvLayer(hidden_dim, 1, 1, 1, is_last=True)


    def forward(self, I):


        en_x4_4 = self.stage4_4(I[3])
        en_x3_3 = self.stage3_3(self.conv_out3(torch.cat([I[2], en_x4_4],1)))
        en_x2_2 = self.stage2_2(self.conv_out2(torch.cat([I[1], en_x3_3],1)))
        en_x1_1 = self.stage1_1(self.conv_out1(torch.cat([I[0], en_x2_2],1)))

        # en_x1_1 = self.conv_out(en_x1_1)
        #
        # en = [en_x1, en_x2, en_x3, en_x4]

        # img_denoise = self.last_1x1(en_x1_1)

        return en_x1_1






def polar_adapt(I, DOP, AOP):
    RO = I * (1 + (9 / 11) * DOP * torch.cos(2 * AOP - 2 * AOP))
    R45 = I * (1 + (9 / 11) * DOP * torch.cos(2 * AOP - 2 * (AOP + math.pi / 4)))
    R90 = I * (1 + (9 / 11) * DOP * torch.cos(2 * AOP - 2 * (AOP + math.pi / 2)))
    R135 = I * (1 + (9 / 11) * DOP * torch.cos(2 * AOP - 2 * (AOP + math.pi * 0.75)))

    return RO, R45, R90, R135


class PD_Block(nn.Module):
    def __init__(self, nb_filter):
        super().__init__()
        self.conv_DOP = ConvLayer(nb_filter, nb_filter, 1, 1)
        self.conv_AOP = ConvLayer(nb_filter, nb_filter, 1, 1)
        self.conv_1 = ConvLayer(nb_filter * 2, nb_filter, 1, 1)

        self.pd_log = pd_log(nb_filter, nb_filter, 1, 1)

    def forward(self, I, DOP, AOP):
        R0, R45, R90, R135 = polar_adapt(I, DOP, AOP)

        DOP_1 = self.conv_DOP(DOP)
        AOP_1 = self.conv_AOP(AOP)
        B_map = self.conv_1(torch.cat((DOP_1, AOP_1), 1))
        B_map = F.softmax(B_map)

        # Ob = torch.log(R0/(R90*R45*R135)) #- torch.log(R90+1e-8) - torch.log(R45+1e-8) - torch.log(R135+1e-8)
        Ob = self.pd_log(R0) - self.pd_log(R45) - self.pd_log(R90) - self.pd_log(R135)

        r0 = R0 * B_map
        r45 = R45 * B_map
        r90 = R90 * B_map
        r135 = R135 * B_map

        # Bg = torch.log(r0/(r90*r45*r135)) # - torch.log(r90+1e-8) - torch.log(r45+1e-8) - torch.log(r135+1e-8)
        Bg = self.pd_log(r0) - self.pd_log(r45) - self.pd_log(r90) - self.pd_log(r135)

        pd = torch.abs(Ob - Bg)  # /(2 * math.log(10))

        return pd


# class SwinTransformer_PD(nn.Module):
#     def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
#                  downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
#         super().__init__()
#
#         self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
#                                   downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
#                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)
#         self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
#                                   downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
#                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)
#         self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
#                                   downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
#                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)
#         self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 4, layers=layers[3],
#                                   downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
#                                   window_size=window_size, relative_pos_embedding=relative_pos_embedding)
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim * 8),
#             nn.Linear(hidden_dim * 8, num_classes)
#         )
#
#         self.Guide1 = Cross_FusionBlock(channels=hidden_dim, hidden_dim=hidden_dim)
#         self.Guide2 = Cross_FusionBlock(channels=hidden_dim * 2, hidden_dim=hidden_dim * 2)
#         self.Guide3 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)
#         self.Guide4 = Cross_FusionBlock(channels=hidden_dim * 4, hidden_dim=hidden_dim * 4)
#
#     # self.dense_encoder=Dense_encoder()
#
#     # self.pd1 = PD_Block(hidden_dim)
#     # self.pd2 = PD_Block(hidden_dim*2)
#     # self.pd3 = PD_Block(hidden_dim*4)
#     # self.pd4 = PD_Block(hidden_dim*4)
#
#     def forward(self, I, PD):
#         en_x1_D = self.stage1(PD)
#         en_x1_D = self.Guide1(en_x1_D, I[0])
#
#         en_x2_D = self.stage2(en_x1_D)
#         en_x2_D = self.Guide2(en_x2_D, I[1])
#
#         en_x3_D = self.stage3(en_x2_D)
#         en_x3_D = self.Guide3(en_x3_D, I[2])
#
#         en_x4_D = self.stage4(en_x3_D)
#         en_x4_D = self.Guide4(en_x4_D, I[3])
#         # PD4 = self.pd4(I[3], en_x4_D, en_x4_A)
#
#         en = [en_x1_D, en_x2_D, en_x3_D, en_x4_D]
#
#         return en


class swin_fusion_net(nn.Module):
    def __init__(self, nb_filter=[32, 64, 128, 128], f_type='res'):
        super().__init__()

        # self.gf_att_map = Cross_FusionModule(in_channels=1, hidden_dimension=32, layers=2,
        #                                      downscaling_factor=1, num_heads=3, head_dim=32,
        #                                      window_size=7, relative_pos_embedding=True)

        # self.gf_1x1 = ConvLayer(2 * 32, 1, 1, 1)


        self.fusion_model = Fusion_network(nb_filter, f_type)


        # self.fusion_model = Cross_SwinT()

        block = DenseBlock_light
        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], 3, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], 3, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], 3, 1)
        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], 3, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], 3, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], 3, 1)

        # self.DB1_1 = StageModule(in_channels=nb_filter[0] + nb_filter[1], hidden_dimension=nb_filter[0])
        #
        # self.DB2_1 = StageModule(in_channels=nb_filter[1] + nb_filter[2], hidden_dimension=nb_filter[1])
        #
        # self.DB3_1 = StageModule(in_channels=nb_filter[2] + nb_filter[3], hidden_dimension=nb_filter[2])
        #
        # self.DB1_2 = StageModule(in_channels=nb_filter[0] * 2 + nb_filter[1], hidden_dimension=nb_filter[0])
        #
        # self.DB2_2 = StageModule(in_channels=nb_filter[1] * 2 + nb_filter[2], hidden_dimension=nb_filter[1])
        #
        # self.DB1_3 = StageModule(in_channels=nb_filter[0] * 3 + nb_filter[1], hidden_dimension=nb_filter[0])


        # self.gf = FastGuidedFilter_attention(r=9, eps=1e-3)
        #
        self.up = nn.Upsample(scale_factor=2)

        self.conv_out = ConvLayer(nb_filter[0], 1, 1, 1, is_last=True)

        self.swin_t_ir = SwinTransformer(
            hidden_dim=32,  # 96
            layers=(2, 2, 2, 2),  # 2,2,6,2
            heads=(3, 3, 3, 3),  # 3, 6, 12, 24
            channels=1,
            num_classes=3,
            head_dim=32,
            window_size=7,
            downscaling_factors=(1, 2, 2, 2),
            relative_pos_embedding=True
        )

        self.swin_up = UP_SwinTransformer()

        self.swin_t_dop = Unet_SwinTransformer(
            hidden_dim=32,  # 96
            layers=(2, 2, 2, 2),  # 2,2,6,2
            heads=(3, 3, 3, 3),  # 3, 6, 12, 24
            channels=1,
            num_classes=3,
            head_dim=32,
            window_size=7,
            scaling_factors=(1, 2, 2, 2),
            relative_pos_embedding=True
        )



    def forward(self, ir,  rho, pd):
        # c_x, c_y = self.gf_att_map(x, pd)
        #
        # att_map = self.gf_1x1(torch.cat((c_x, c_y), 1))
        #
        # # att_map = F.relu(theta_x + phi_g, inplace=True)
        #
        # att_map = F.sigmoid(att_map)
        #
        # y_gf = self.gf(x, y, att_map)

        # dop_ir = torch.cat([dop,ir],1)

        # en_dop = self.swin_t_dop(dop)
        en_ir = self.swin_t_ir(ir)

        en_AGF = self.swin_t_dop(rho, pd, en_ir)

        # en_vi = self.swin_t_y(y_gf)

        # en_ir = self.dense_encoder(x)
        # en_vi = self.dense_encoder(y)

        # out0 = torch.cat([en_ir[0], en_vi[0]], 1)
        # out1 = torch.cat([en_ir[1], en_vi[1]], 1)
        # out2 = torch.cat([en_ir[2], en_vi[2]], 1)
        # out3 = torch.cat([en_ir[3], en_vi[3]], 1)
        #
        # f_en0 = self.conv_f0(out0)
        # f_en1 = self.conv_f1(out1)
        # f_en2 = self.conv_f2(out2)
        # f_en3 = self.conv_f3(out3)
        #
        # f_en = [f_en0,f_en1,f_en2,f_en3]





        # f_en = self.fusion_model(en_ir, en_AGF)






        # out = f_en[3].detach().cpu().numpy().squeeze()
        #
        # for i in range(16):
        #     plt.subplot(4, 4, i + 1)
        #     plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
        #     plt.imshow(out[i, :, :])
        #     plt.axis('off')
        # plt.show()

        # plt.imshow(out[5, :, :])
        # plt.show()

        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))
        #
        # x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        # x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))
        #
        # x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        # x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        #
        # x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        output = self.swin_up(en_AGF)

        output = self.conv_out(output)
        # en_output = self.swin_t_x(output)

        return output  # , y_gf #, y_gf, en_output, f_en

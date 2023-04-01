import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from imageio import imread, imsave
from fusion import AFF, iAFF, DAF


class NoisyDataset(nn.Module):
    def __init__(self, mean=0, var=.5):
        super(NoisyDataset, self).__init__()
        self.mean = mean
        self.var = var

    def forward(self, image):
        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        noise = np.random.normal(self.mean, self.var, size=image.shape)
        noisy_image = np.clip(image + noise, low_clip, 1)
        return noisy_image


def subsample(image, k=2):
    # This function only works for k = 2 as of now.
    blen, channels, m, n = np.shape(image)
    dim1, dim2 = m // k, n // k
    image1, image2 = np.zeros([blen, channels, dim1, dim2]), np.zeros([blen, channels, dim1, dim2])
    upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    image = image.cpu()
    for channel in range(channels):
        for i in range(dim1):
            for j in range(dim2):
                i1 = i * k
                j1 = j * k
                num = np.random.choice([0, 1, 2, 3])
                if num == 0:
                    image1[:, channel, i, j], image2[:, channel, i, j] = image[:,
                                                                         channel, i1, j1], image[:, channel, i1, j1 + 1]
                elif num == 1:
                    image1[:, channel, i, j], image2[:, channel, i, j] = image[:,
                                                                         channel, i1 + 1, j1], image[:, channel, i1 + 1,
                                                                                               j1 + 1]
                elif num == 2:
                    image1[:, channel, i, j], image2[:, channel, i, j] = image[:,
                                                                         channel, i1, j1], image[:, channel, i1 + 1, j1]
                else:
                    image1[:, channel, i, j], image2[:, channel, i, j] = image[:,
                                                                         channel, i1, j1 + 1], image[:, channel, i1 + 1,
                                                                                               j1 + 1]

    # if self.use_cuda:
    #     return upsample(torch.from_numpy(image1)).cuda(), upsample(torch.from_numpy(image2)).cuda()

    # F.interpolate()
    return torch.from_numpy(image1).cuda(), torch.from_numpy(image2).cuda()


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class MixedFusion_Block(nn.Module):

    def __init__(self):
        super(MixedFusion_Block, self).__init__()

        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=1, stride=1, padding=1),
        #                             nn.BatchNorm2d(in_dim), act_fn, )

        # revised in 09/09/2019.
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        # self.layer2 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=1),
        #                             nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, x1, x2):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        # out1 = self.layer1(out_fusion)
        # out2 = self.layer2(torch.cat((out1, xx), dim=1))

        return out_fusion


class fusion_sf(nn.Module):
    def __init__(self, channels):
        super(fusion_sf, self).__init__()
        self.kernel_radius = 5
        self.GRF = Sobelxy(channels)

    def forward(self, f1, f2):
        """
        Perform channel sf fusion two features
        """
        # device = f1.device

        # f1 = self.GRF(f1)
        # f2 = self.GRF(f2)

        b, c, h, w = f1.shape

        r_shift_kernel = torch.DoubleTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) \
            .cuda().reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.DoubleTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) \
            .cuda().reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = F.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = F.conv2d(f2, b_shift_kernel, padding=1, groups=c)

        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

        kernel_size = self.kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).double().cuda()
        kernel_padding = kernel_size // 2
        f1_sf = torch.sum(F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        f2_sf = torch.sum(F.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)

        modal_in1 = f1_sf.unsqueeze(1)
        modal_in2 = f2_sf.unsqueeze(1)
        #
        # fusion_n = torch.abs(modal_in1-modal_in2)/(modal_in1+modal_in2)

        # fusion_map = torch.relu(fusion_map)
        # fusion_map = torch.sigmoid(fusion_n).cuda()

        weight_zeros = torch.zeros(modal_in2.shape).double().cuda()
        weight_ones = torch.ones(modal_in2.shape).double().cuda()
        #
        fusion_map = torch.where(modal_in1 < modal_in2, weight_ones, weight_zeros).cuda()
        # dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
        # out = fusion_map.detach().cpu().numpy().squeeze()
        # imsave('test_results' + '/fusion_map.png', out.astype('uint8'))

        return fusion_map


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        # self.conv_fusion = ConvLayer(1, channels, ws[index], 1)
        #
        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        # block1 = []
        # block1 += [ConvLayer(2 * channels, channels, 1, 1),
        #            ConvLayer(channels, channels, ws[index], 1)]
        # self.bottelblock1 = nn.Sequential(*block1)
        #
        # block2 = []
        # block2 += [ConvLayer(2 * channels, channels, 1, 1),
        #            ConvLayer(channels, channels, ws[index], 1)]
        # self.bottelblock2 = nn.Sequential(*block2)
        # #
        # self.GRF=Sobelxy(2*channels)
        #
        # # self.SF = fusion_sf(channels)
        # #
        # # self.Mix = MixedFusion_Block()

        self.iAFF = iAFF(channels)

        # self.AFF = AFF(channels)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')


        # f_cat = torch.cat([x_ir, x_vi], 1)
        # f_GRF = self.GRF(f_cat)
        # f_GRF = self.bottelblock1(f_GRF)
        # # out = self.AFF(f_iAFF, f_GRF)
        #
        # # f_SF = self.SF(x_ir, x_vi)
        # # SF = torch.mul(x_vi, f_SF)
        # #
        # # # f_init = self.conv_fusion(f_cat)
        # #
        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)

        f_iAFF = self.iAFF(out_ir, out_vi)
        # # # out_mix = self.Mix(out_ir,out_vi)
        # #
        # out_mix = torch.cat([out_ir, out_vi], 1)
        # out = self.bottelblock2(out_mix)
        # # out = self.AFF(out, f_GRF)
        # # out = torch.cat([out, SF], 1)
        # # out = self.bottelblock2(out)
        #
        # out = f_GRF + out


        return f_iAFF


# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type

        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


class Dense_encoder(nn.Module):
    def __init__(self, nb_filter=[32, 64, 128, 128], input_nc=1, output_nc=1):
        super(Dense_encoder, self).__init__()
        # self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2)
        # self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def forward(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        # n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        # hr_x = hr_x.double()
        l_a = l_a.double()

        # assert n_lrx == n_lry and n_lry == n_hrx
        # assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        # assert h_lrx == h_lry and w_lrx == w_lry
        # assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        # mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        # mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (A * lr_x + b).float()

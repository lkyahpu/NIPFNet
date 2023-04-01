from math import exp
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
from demo_score import main
from inference_iqa_en import IQA,EN
#import TVLoss
from cv2.ximgproc import guidedFilter
from torchvision.transforms import ToPILImage


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map, sigma1_sq


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    value, sigma1_sq = _ssim(img1, img2, window, window_size, channel, size_average)
    v = torch.zeros_like(sigma1_sq) + 0.0001
    sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
    return value, sigma1


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.DoubleTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.DoubleTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient




class L_Grad_single(nn.Module):
    def __init__(self):
        super(L_Grad_single, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A,  image_fused):
        gradient_A = self.sobelconv(image_A)

        gradient_fused = self.sobelconv(image_fused)

        Loss_gradient = F.l1_loss(gradient_fused, gradient_A)

        return Loss_gradient



class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        # intensity_joint = torch.max(intensity_joint, image_C)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


def func_loss(img1, img2, y):
    #img1, img2 = tf.split(y_, 2, 3)
    img3 = img1 * 0.5 + img2 * 0.5
    Win = [11, 9, 7, 5, 3]
    loss = 0
    for s in Win:
        loss1, sigma1 = ssim(img1, y, s)
        loss2, sigma2 = ssim(img2, y, s)
        r = sigma1 / (sigma1 + sigma2 + 0.0000001)
        tmp = 1 - torch.mean(r * loss1) - torch.mean((1 - r) * loss2)
        loss = loss + tmp
    loss = loss / 5.0
    loss = loss + torch.mean(torch.abs(img3 - y)) * 0.1
    return loss



class pf_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_Grad1 = L_Grad()
        self.L_Grad2 = L_Grad()
        self.TV = TVLoss()
        self.L_Inten = L_Intensity()

    def forward(self, y1, y2, y):


        loss_mssim = func_loss(y1, y2, y)

        # loss = loss + 0.1*func_loss(y1, pd, p_gf)
        #
        loss_Intensity = self.L_Inten(y1, y2, y)
        #
        # loss = loss + 0.1 * loss_Intensity

        loss_gradient = self.L_Grad1(y1, y2, y)

        # loss = loss + 0.1*0.25*(self.TV(x1[0]-x2[0])+self.TV(x1[1]-x2[1])+self.TV(x1[2]-x2[2])+self.TV(x1[3]-x2[3]))
        # loss = loss + 0.1*loss_gradient
        # loss = loss + 0.1*self.L_Grad2(y1, y2, p_gf)
        #loss_aop = torch.mean(torch.abs(aop_t - aop))
        #loss_p = 0.5 * loss_dop + 0.5 * loss_aop
        loss = loss_mssim #+ 0.1*loss_gradient + 0.1*loss_Intensity

        return loss



class VGG_percept_loss(nn.Module):
    def __init__(self):
        super(VGG_percept_loss, self).__init__()
        vgg19_model = vgg19(pretrained=True).double()
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.feature_extractor.eval()
        for parm in self.feature_extractor.parameters():
            parm.requires_grad = False
        self.p_criterion = nn.L1Loss()

    def forward(self, x, y):

        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        x_feature = self.feature_extractor(x)
        y_feature = self.feature_extractor(y)
        loss = self.p_criterion(x_feature, y_feature)

        return loss



class RegularizedLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()

        self.gamma = gamma
        self.L_Grad1 = L_Grad()
        self.L_Inten = L_Intensity()
        self.mae = nn.L1Loss()
        # self.percept_loss = VGG_percept_loss()
        self.TVLoss = TVLoss()
        self.sobelconv = Sobelxy()
        self.L_Grad_single = L_Grad_single()

    def gaussian(self,window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self,window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map, sigma1_sq

    def ssim(self, img1, img2, window_size=11, size_average=True):
        img1 = torch.clamp(img1, min=0, max=1)
        img2 = torch.clamp(img2, min=0, max=1)
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        value, sigma1_sq = self._ssim(img1, img2, window, window_size, channel, size_average)
        v = torch.zeros_like(sigma1_sq) + 0.0001
        sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
        return value, sigma1

    def mseloss(self, image, target):
        x = ((image - target)**2)
        return torch.mean(x)

    def regloss(self, g1, g2, G1, G2):
        return torch.mean((g1-g2-G1+G2)**2)





    def mssim_loss(self, img1, img2,img):
        # img1, img2 = tf.split(y_, 2, 3)
        img3 = img1 * 0.5 + img2 * 0.5
        Win = [11, 9, 7, 5, 3]
        loss = 0


        for s in Win:
            loss1, sigma1 = self.ssim(img1, img, s)
            loss2, sigma2 = self.ssim(img2, img, s)

            r = sigma1 / (sigma1 + sigma2 + 0.0000001)
            tmp = 1 - torch.mean(r * loss1) - torch.mean((1 - r) * loss2)
            # tmp = 1 - w1*torch.mean(loss1) - w2*torch.mean(loss2)

            loss = loss + tmp
        loss = loss / 5.0
        loss = loss + torch.mean(torch.abs(img3 - img)) * 0.1
        return loss

    def mssim3_loss(self, img1, img2, img3, img):
        # img1, img2 = tf.split(y_, 2, 3)
        img4 = (img1 + img2 + img3)/3.0
        Win = [11, 9, 7, 5, 3]
        loss = 0


        for s in Win:
            loss1, sigma1 = self.ssim(img1, img, s)
            loss2, sigma2 = self.ssim(img2, img, s)
            loss3, sigma3 = self.ssim(img3, img, s)

            r1 = sigma1 / (sigma1 + sigma2 + sigma3 + 0.0000001)
            r2 = sigma2 / (sigma1 + sigma2 + sigma3 + 0.0000001)
            r3 = sigma3 / (sigma1 + sigma2 + sigma3 + 0.0000001)

            tmp = 1 - torch.mean(r1 * loss1) - torch.mean(r2 * loss2) - torch.mean(r3 * loss3)
            # tmp = 1 - w1*torch.mean(loss1) - w2*torch.mean(loss2)

            loss = loss + tmp
        loss = loss / 5.0
        # loss = loss + torch.mean(torch.abs(img4 - img)) * 0.1
        return loss

    def grad(self, image_A, image_B, image_C,image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_C = self.sobelconv(image_C)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        gradient_joint = torch.max(gradient_joint, gradient_C)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

    # def forward(self, fg1, g2, g1f, g2f, img_ir, X, output):
    #     loss_all = self.mseloss(fg1, g2) + self.gamma * self.regloss(fg1, g2, g1f, g2f) + self.mssim_loss(img_ir, X, output)
    #     return loss_all


    def forward(self, img_ir, img_p, img_pd, output_pf):


        p_loss = self.mssim3_loss(img_ir, img_p, img_pd, output_pf)

        # p_grad_loss = self.grad(img_ir, img_p, img_pd, output_pf)
        p_grad_loss = self.mssim3_loss(self.sobelconv(img_ir), self.sobelconv(img_p), self.sobelconv(img_pd), self.sobelconv(output_pf))

        pd_grad = self.L_Grad1(img_p, img_pd, output_pf)



        loss_all = p_loss + 0.5*p_grad_loss + 0.1*pd_grad     #0.5*p_loss + p_grad_loss +  + p_Inten_loss


        return loss_all   #self.mssim_loss(img_ir, img_p, output_pf)


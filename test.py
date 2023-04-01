import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from dataloader import list_images, load_dataset, get_train_images, get_image
from net import pf_net
import loss_func
from torch.utils.tensorboard import SummaryWriter
from imageio import imread, imsave
import random
import shutil
from torchvision import transforms
from swin_fusion import swin_fusion_net

EPSILON = 1e-4
import numpy as np
from torchvision import transforms
from function import NoisyDataset, subsample


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)





def test(image_set_ir):

    #criterion = loss_func.pf_loss().cuda()
    #image_set_ir, batches = load_dataset(data_iter, 32)
    #pfnet.eval()
    #pfnet.cuda()
    # noisy = NoisyDataset()
    count = 0
    transform = transforms.Compose(
        [ transforms.Resize((448,448)), transforms.ToTensor()])   #transforms.CenterCrop(448), transforms.Resize((448,448))


    #image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]

    img_ir = get_image(image_set_ir)
    img_ir = transform(img_ir)


    # img_ir = torch.from_numpy(np.expand_dims(np.expand_dims(img_ir, axis=0), axis=0)).float()

    image_set_p = image_set_ir.replace('S0', 'rho')
    img_p = get_image(image_set_p)
    img_p = transform(img_p)

    image_set_pd = image_set_ir.replace('S0', 'pd')
    img_pd = get_image(image_set_pd)
    img_pd = transform(img_pd)

    # image_set_dolp = image_set_ir.replace('S0', 'dolp')
    # img_dolp = get_image(image_set_dolp)
    # img_dolp = transform(img_dolp)

    # img_p = noisy(img_p)


    # img_p = torch.from_numpy(np.expand_dims(np.expand_dims(img_p, axis=0), axis=0)).float()

    # image_set_pd = image_set_ir.replace('S0', 'pd')
    # img_pd = get_image(image_set_pd, height=448, width=448)
    # img_pd = torch.from_numpy(np.expand_dims(np.expand_dims(img_pd, axis=0), axis=0)).float()


    # img_p = get_train_images(image_paths_p, height=448, width=448)

    # image_paths_pd = [x.replace('S0', 'pd') for x in image_set_ir]
    # img_pd = get_train_images(image_paths_pd, height=448, width=448)


    # img_ir = (img_ir - torch.min(img_ir)) / (torch.max(img_ir) - torch.min(img_ir) + EPSILON)
    # img_p = (img_p - torch.min(img_p)) / (torch.max(img_p) - torch.min(img_p) + EPSILON)
    # img_pd = (img_pd - torch.min(img_pd)) / (torch.max(img_pd) - torch.min(img_pd) + EPSILON)

    img_ir = img_ir.cuda()
    img_p = img_p.cuda()
    img_pd = img_pd.cuda()
    # img_dolp = img_dolp.cuda()

    #img_ir = img_ir[:,:,64:,96:-96]
    #img_p = img_p[:,:,64:,96:-96]

    #output_pf = pfnet(img_ir, img_p)

    #loss = criterion(img_ir, img_p, output_pf)


    return img_ir.double(), img_p.double(), img_pd.double()


if __name__ == "__main__":

    dir_path = 'test/'
    test_path = 'dataset/'+dir_path
    pfnet = swin_fusion_net().double()
    pfnet.cuda()

    #pfnet = pf_net()
    pfnet.load_state_dict(torch.load('save_pth/Epoch.pth'))
    #pfnet.cuda()
    pfnet.eval()
    test_imgs_path = list_images(test_path)
    test_imgs_path = [x.replace('\\', '/') for x in test_imgs_path]
    mk_path = 'test_results/' + dir_path + 'results/'
    mkdir(mk_path)
    # img_ir, img_p = test(test_imgs_path)
    # s=img_ir[0].unsqueeze(1).shape
    with torch.no_grad():
      for test_dir in test_imgs_path:
       img_ir, img_p, img_pd = test(test_dir)
       img_ir = img_ir.unsqueeze(0)
       img_p = img_p.unsqueeze(0)
       img_pd = img_pd.unsqueeze(0)
       # img_dolp = img_dolp.unsqueeze(0)

       try:
           # output_pf = pfnet(img_ir, img_p, img_pd)
           output_pf = pfnet(img_ir, img_p)
       except RuntimeError as exception:
           if "out of memory" in str(exception):
               print("WARNING: out of memory")
               if hasattr(torch.cuda, 'empty_cache'):
                   torch.cuda.empty_cache()
           else:
               raise exception

       # torchvision.utils.save_image(output_pf, 'test_results/'+test_dir.split('/')[-2]+'.png')
       out = output_pf.detach().cpu().numpy().squeeze()
       out = (out - np.min(out)) / (np.max(out) - np.min(out))

       img_ir = img_ir.detach().cpu().numpy().squeeze()
       # img_ir = np.concatenate((np.expand_dims(img_ir,axis=-1),np.expand_dims(img_ir,axis=-1),np.expand_dims(img_ir,axis=-1)),axis=-1)
       img_ir = (img_ir - np.min(img_ir)) / (np.max(img_ir) - np.min(img_ir))

       img_p = img_p.detach().cpu().numpy().squeeze()
       img_p = (img_p - np.min(img_p)) / (np.max(img_p) - np.min(img_p))

       img_pd = img_pd.detach().cpu().numpy().squeeze()
       img_pd = (img_pd - np.min(img_pd)) / (np.max(img_pd) - np.min(img_pd))

       # img_dolp = img_dolp.detach().cpu().numpy().squeeze()
       # img_dolp = np.concatenate((np.expand_dims(img_dolp,axis=-1), np.expand_dims(img_dolp,axis=-1), np.expand_dims(img_dolp,axis=-1)), axis=-1)
       # img_dolp = (img_dolp - np.min(img_dolp)) / (np.max(img_dolp) - np.min(img_dolp))


       scene_n = test_dir.split('/')[-2]
       mkdir( mk_path + scene_n)
       scene_dir = mk_path + scene_n

       # imsave(scene_dir + '/S0.png', (img_ir * 255).astype('uint8'))
       # imsave(scene_dir + '/pd.png', (img_pd * 255).astype('uint8'))
       # imsave(scene_dir + '/dolp.png', (img_p * 255).astype('uint8'))
       # imsave(scene_dir + '/dolp.png', (img_dolp * 255).astype('uint8'))

       # denoise = denoise.detach().cpu().numpy().squeeze()
       # denoise = (denoise - np.min(denoise)) / (np.max(denoise) - np.min(denoise))

       imsave(scene_dir + '/fusion.png', (out*255).astype('uint8'))
       # imsave('gf_results/' + test_dir.split('/')[-2] + '.png', (denoise * 255).astype('uint8'))
       print(test_dir.split('/')[-2])
       # break

    print('finished!!!')

    # torchvision.utils.save_image(out_pf, test_path.replace('ir', 'result') + test_imgs_path[0].split('/')[-1])
    # print(out_pf.shape)














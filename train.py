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
from dataloader import list_images, load_dataset, get_train_images
from cv2.ximgproc import guidedFilter
import net
import loss_func
from torch.utils.tensorboard import SummaryWriter
from swin_fusion import swin_fusion_net
from function import NoisyDataset, subsample
from imageio import imread
import random
import shutil
import pickle
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

EPSILON = 1e-4
import numpy as np
from torchvision import transforms

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# torch.cuda.set_device(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
writer = SummaryWriter('log/')
start_time = time.time()

train_loss_list = []
test_loss_list = []


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.fill_(0)


# class polar_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, dop_t,  dop):
#         loss_dop = torch.mean(torch.abs(dop_t - dop))
#         #loss_aop = torch.mean(torch.abs(aop_t - aop))
#         #loss_p = 0.5 * loss_dop + 0.5 * loss_aop
#         return loss_dop


def evaluate_loss(data_iter, pfnet):
    batch_size = 4
    criterion = loss_func.RegularizedLoss().to(device)
    image_set_ir, batches = load_dataset(data_iter, batch_size)

    transform = transforms.Compose(
        [transforms.Resize((config.HEIGHT, config.WIDTH)),transforms.ToTensor()])
    pfnet.eval()
    # pfnet.to('cpu')
    count = 0
    test_loss_sum = 0.0
    noisy = NoisyDataset()
    with torch.no_grad():
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]

            img_ir = get_train_images(image_paths_ir, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            image_paths_p = [x.replace('S0', 'rho') for x in image_paths_ir]
            img_p = get_train_images(image_paths_p, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            image_paths_pd = [x.replace('S0', 'pd') for x in image_paths_ir]
            img_pd = get_train_images(image_paths_pd, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            img_ir_origin = get_train_images(image_paths_ir, flag=config.img_flag)
            image_paths_p = [x.replace('S0', 'rho') for x in image_paths_ir]
            img_p_origin = get_train_images(image_paths_p, flag=config.img_flag)

            image_paths_pd = [x.replace('S0', 'pd') for x in image_paths_ir]
            img_pd_origin = get_train_images(image_paths_pd, flag=config.img_flag)

            # img_p = noisy(img_p)

            # g1, g2 = subsample(img_p)
            # ir1, ir2 = subsample(img_ir)


            images_p_g = []
            for b in range(np.array(img_ir).shape[0]):
                img_p_g = guidedFilter(np.array(img_ir_origin)[b, :, :, :].squeeze(),
                                       np.array(img_p_origin)[b, :, :, :].squeeze(), 11,
                                       1e-6)

                img_p_g = Image.fromarray(img_p_g)
                # img_p_g = img_p_g.resize((config.HEIGHT, config.WIDTH))
                # img_p_g = np.array(img_p_g)
                # plt.figure('image')
                # plt.imshow(img_p_g, cmap='gray')
                # plt.show()
                img_p_g = transform(img_p_g)
                images_p_g.append(img_p_g)



            images_pd_g = []
            for b in range(np.array(img_ir).shape[0]):
                img_pd_g = guidedFilter(np.array(img_ir_origin)[b, :, :, :].squeeze(),
                                       np.array(img_pd_origin)[b, :, :, :].squeeze(), 11,
                                       1e-6)

                img_pd_g = Image.fromarray(img_pd_g)
                # img_p_g = img_p_g.resize((config.HEIGHT, config.WIDTH))
                # img_p_g = np.array(img_p_g)
                # plt.figure('image')
                # plt.imshow(img_p_g, cmap='gray')
                # plt.show()
                img_pd_g = transform(img_pd_g)
                images_pd_g.append(img_pd_g)




            images_p_g = torch.stack(images_p_g, axis=0)
            images_p_g = images_p_g.to(device)
            images_p_g = images_p_g.double()

            images_pd_g = torch.stack(images_pd_g, axis=0)
            images_pd_g = images_pd_g.to(device)
            images_pd_g = images_pd_g.double()


            img_p = img_p.to(device)
            img_p = img_p.double()

            img_pd = img_pd.to(device)
            img_pd = img_pd.double()

            img_ir = img_ir.to(device)
            img_ir = img_ir.double()

            # img_ir = (img_ir - torch.min(img_ir)) / (torch.max(img_ir) - torch.min(img_ir) + EPSILON)
            # img_p = (img_p - torch.min(img_p)) / (torch.max(img_p) - torch.min(img_p) + EPSILON)

            #output_pf = pfnet(img_ir,img_p)
            output_pf = pfnet(img_ir, img_p, img_pd)
            
            # pd_n, _ = pfnet(img_ir, img_pd)
            # G1, G2 = subsample(X)

            # loss = criterion(img_ir, img_p, output_pf)
            # loss = criterion(p_n, img_pd, pd_n, img_p, img_ir, p_n, output_pf)
            
            # loss = criterion(img_ir, images_p_g, images_pd_g, output_pf)
            loss = criterion(img_ir, img_p, img_pd, output_pf)
            

            test_loss_sum += loss.cpu().item()

            count += 1
        # break

    return test_loss_sum / count


def train(config):
    # device = torch.device('cpu')
    start_time = time.time()
    # test_n = [74, 77, 209, 279]
    batch_size = config.train_batch_size
    # batch_count = 0.0
    transform = transforms.Compose(
        [transforms.Resize((config.HEIGHT, config.WIDTH)),transforms.ToTensor()])

    train_imgs_path = list_images(config.train_images_path)
    val_imgs_path = list_images(config.val_images_path)

    # random.shuffle(original_imgs_path)
    #
    # num = len(original_imgs_path)
    #
    # num_s = (num // 4) * 3

    train_set = train_imgs_path  # [:num_s]

    val_set = val_imgs_path  # [num_s:]

    loss_min = 100.0
    pfnet = swin_fusion_net().double()  # net.psa_net()
    pfnet = pfnet.to(device)

    # pfnet.apply(weights_init)
    # pfnet.load_state_dict(torch.load('save_pth/Epoch.pth',map_location=device))


    criterion = loss_func.RegularizedLoss().to(device)

    optimizer = torch.optim.Adam(pfnet.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    noisy = NoisyDataset()

    # dehaze_net.train()
    # val_loss_min = 1.0

    for epoch in range(config.num_epochs):

        # load training database
        image_set_ir, batches = load_dataset(train_set, batch_size)
        pfnet.train()
        pfnet.to(device)
        count = 0
        train_loss_sum = 0.0

        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]

            img_ir = get_train_images(image_paths_ir, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            image_paths_p = [x.replace('S0', 'rho') for x in image_paths_ir]
            img_p = get_train_images(image_paths_p, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            image_paths_pd = [x.replace('S0', 'pd') for x in image_paths_ir]
            img_pd = get_train_images(image_paths_pd, height=config.HEIGHT, width=config.WIDTH, flag=config.img_flag)

            img_ir_origin = get_train_images(image_paths_ir, flag=config.img_flag)
            image_paths_p = [x.replace('S0', 'rho') for x in image_paths_ir]
            img_p_origin = get_train_images(image_paths_p, flag=config.img_flag)

            image_paths_pd = [x.replace('S0', 'pd') for x in image_paths_ir]
            img_pd_origin = get_train_images(image_paths_pd, flag=config.img_flag)

            # img_p = noisy(img_p)

            # g1, g2 = subsample(img_p)
            # ir1, ir2 = subsample(img_ir)

            images_p_g = []
            for b in range(np.array(img_ir).shape[0]):
                img_p_g = guidedFilter(np.array(img_ir_origin)[b, :, :, :].squeeze(),
                                       np.array(img_p_origin)[b, :, :, :].squeeze(), 11,
                                       1e-6)

                img_p_g = Image.fromarray(img_p_g)
                # img_p_g = img_p_g.resize((config.HEIGHT, config.WIDTH))
                # img_p_g = np.array(img_p_g)
                # plt.figure('image')
                # plt.imshow(img_p_g, cmap='gray')
                # plt.show()
                img_p_g = transform(img_p_g)
                images_p_g.append(img_p_g)

            images_pd_g = []
            for b in range(np.array(img_ir).shape[0]):
                img_pd_g = guidedFilter(np.array(img_ir_origin)[b, :, :, :].squeeze(),
                                        np.array(img_pd_origin)[b, :, :, :].squeeze(), 11,
                                        1e-6)

                img_pd_g = Image.fromarray(img_pd_g)
                # img_p_g = img_p_g.resize((config.HEIGHT, config.WIDTH))
                # img_p_g = np.array(img_p_g)
                # plt.figure('image')
                # plt.imshow(img_p_g, cmap='gray')
                # plt.show()
                img_pd_g = transform(img_pd_g)
                images_pd_g.append(img_pd_g)

            images_p_g = torch.stack(images_p_g, axis=0)
            images_p_g = images_p_g.to(device)
            images_p_g = images_p_g.double()

            images_pd_g = torch.stack(images_pd_g, axis=0)
            images_pd_g = images_pd_g.to(device)
            images_pd_g = images_pd_g.double()

            img_p = img_p.to(device)
            img_p = img_p.double()

            img_ir = img_ir.to(device)
            img_ir = img_ir.double()

            img_pd = img_pd.to(device)
            img_pd = img_pd.double()



            output_pf = pfnet(img_ir, img_p, img_pd)
            #output_pf = pfnet(img_ir, img_p)

            # with torch.no_grad():
            #     pd_n, _ = pfnet(img_ir, img_pd)
            # G1, G2 = subsample(X)

            # loss = criterion(img_ir, img_p, output_pf)
            # loss = criterion(p_n, img_pd, pd_n, img_p, img_ir, p_n, output_pf)
            
            loss = criterion(img_ir, img_p, img_pd, output_pf)
            # loss = criterion(img_ir, images_p_g, images_pd_g, output_pf)
            

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            count += 1
            train_loss_list.append(loss.item())
            # print("iter", count, 'train_loss: %.3f' % (loss.cpu().item()),end='')
            print(
                f'\repoch:{epoch + 1}| iter:{count}/{batches}|train loss : {loss.item():.3f}|time_used :{(time.time() - start_time) / 60 :.1f}',
                end='', flush=True)
            if count % 50 == 0:

                # epoch_loss = train_loss_sum / count

                test_loss = evaluate_loss(val_set, pfnet)
                test_loss_list.append(test_loss)
                pfnet.to(device)
                # writer.add_scalar('Loss/train_st', loss.item(), epoch + 1)
                # writer.add_scalar('Loss/val_st', test_loss, epoch + 1)

                # print('\n',"Epoch", epoch + 1, 'train_loss: %.3f' % (loss.item()), 'test_loss: %.3f' % (test_loss))
                # print(
                #     f'\repoch:{epoch + 1}| iter:{count}/{batches}|train loss: {loss.item():.3f}|test_loss: {test_loss:.3f}|time_used :{(time.time() - start_time) / 60 :.1f}',
                #     end='', flush=True)

                if test_loss < loss_min:
                    loss_min = test_loss
                    torch.save(pfnet.state_dict(),
                               config.save_pths_folder + "Epoch_" + str(epoch + 1) + 'pf_net_loss%.3f' % (
                                   test_loss) + '.pth')
                    print('test_loss%.3f' % (test_loss) + 'model saved!!!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--train_images_path', type=str, default="dataset/train/")
    parser.add_argument('--val_images_path', type=str, default="dataset/val/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--save_pths_folder', type=str, default="save_pth/")
    parser.add_argument('--sample_output_folder', type=str, default="test_results/")
    parser.add_argument('--HEIGHT', type=int, default=112)
    parser.add_argument('--WIDTH', type=int, default=112)
    parser.add_argument('--img_flag', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.save_pths_folder):
        os.mkdir(config.save_pths_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)

    # pass
    # test_set = original_imgs_path[num_s:]

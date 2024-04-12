# -- coding:utf-8 --
#train prediction and super-resolution separately
import argparse
import os
import numpy as np
from math import log10,sqrt
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data
from torch import nn
from datetime import datetime
import pytorch_ssim
from data_process_pre import get_dataloader_pre,get_dataloader_sr
from prediction import TransAm
from sr import Generator,Discriminator
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--dataset', type=str, default='P4', help='which dataset to use')
# 1500 和 100:根据统计数据后粗细粒度每个格子大概的信号容量制定
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
#prediction
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=3)
parser.add_argument('--len_trend', type=int, default=0)
parser.add_argument('--external_dim', type=int, default=7)
parser.add_argument('--n_heads', type=int, default=2,
                    help='number of heads of selfattention')
parser.add_argument('--dim_head', type=int, default=8,
                    help='dim of heads of selfattention')
parser.add_argument('--dropout', type=float, default=0,
                    help='encoder dropout')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of encoder layers')
parser.add_argument('--feature_size', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dim of FC layer')
parser.add_argument('--skip_dim', type=int, default=256,
                    help='dim of skip conv')

# training skills
parser.add_argument('--lr', type=float, default=1e-3, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--harved_epoch', type=int, default=20, help='halved at every x interval')
parser.add_argument('--seed', type=int, default=2023, help='random seed')

def get_RMSE(pred, real):
    mse = np.mean(np.power(real - pred, 2))
    return sqrt(mse)

def train_pre(lr,epoch_num):
    # 设置神经网络参数随机初始化种子，使每次训练初始参数可控
    torch.cuda.manual_seed(opt.seed)
    rmses = [np.inf]
    save_path = 'saved_model/separate/{}/cpt/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
                                                               opt.len_closeness,
                                                               opt.len_period,
                                                               opt.len_trend,
                                                               opt.n_heads,
                                                               opt.num_layers,
                                                                opt.skip_dim)
    os.makedirs(save_path, exist_ok=True)
    datapath = os.path.join('data', opt.dataset)

    train_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  opt.batch_size,True,
        mode='train')  # opt.batch_size=16
    valid_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  4, True,mode='valid')

    pre = TransAm(feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           external_dim=opt.external_dim, dropout=opt.dropout)
    print('# prediction parameters:', sum(param.numel() for param in pre.parameters()))


    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        print("CUDA可用，正在用GPU运行程序")
        pre.cuda()
        criterion.cuda()

    optimizer = optim.Adam(pre.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    iter = 0
    valid_rmse = torch.zeros((epoch_num,))
    for epoch in range(epoch_num):
        pre.train()  # model.train()：启用Batch_Normalization和Dropout
        train_loss = 0
        ep_time = datetime.now()
        """生成样本数据"""
        for z, (xc,xp,xt,ext,next) in enumerate(train_dataloader):

            optimizer.zero_grad()
            loss = 0
            B,Tc,H,W = xc.shape
            if torch.cuda.is_available():
                xc = xc.cuda()
                xp = xp.cuda()
                xt = xt.cuda()
                ext = ext.cuda()
                next = next.cuda()
            pred = pre(xc,xp,xt,ext)
            loss = criterion(pred, next.reshape(B,1,H,W))
            loss.requires_grad_(True)
            # 更新网络参数
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                    epoch_num,
                                                                    z,
                                                                    len(train_dataloader),
                                                                    np.sqrt(loss.item())
                                                                    ))

            # counting training mse
            train_loss += loss.item()

            iter += 1
            # validation phase
            if iter % 20 == 0:
                with torch.no_grad():
                    pre.eval()
                    valid_time = datetime.now()
                    total_mse = 0
                    for n, (xc, xp, xt, ext, next) in enumerate(valid_dataloader):
                        los = 0
                        if torch.cuda.is_available():
                            xc = xc.cuda()
                            xp = xp.cuda()
                            xt = xt.cuda()
                            ext = ext.cuda()
                        Bv, Tv, H, W = xc.shape
                        pred = pre(xc, xp, xt, ext).cpu()
                        # 计算和预期输出之间的MSE损失
                        los = criterion(pred, next.reshape(Bv, 1,H, W))
                        total_mse += los * Bv
                    rmse = np.sqrt(total_mse / len(valid_dataloader.dataset)) * opt.scaler_X
                    valid_rmse[epoch]=rmse
                    if rmse < np.min(rmses):
                        print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now() - valid_time))
                        torch.save(pre.state_dict(),
                                   '{}/final_model.pt'.format(save_path))
                    rmses.append(rmse)

        # half the learning rate
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            lr /= 2
            optimizer = optim.Adam(pre.parameters(), lr=lr)

        print('=================time cost: {}==================='.format(
            datetime.now() - ep_time))
    np.save(os.path.join(save_path, 'valid_rmse.npy'), valid_rmse.numpy())
    return

def train_sr(lr,epoch_num):
    torch.cuda.manual_seed(opt.seed)
    rmses = [np.inf]
    g_save_path = 'saved_model/separate/{}/SR/{}-{}-{}_{}{}{}_{}_{}_{}/Generator'.format(opt.dataset,
                                                                             opt.n_residuals,
                                                                             opt.base_channels,
                                                                             opt.num_epochs,
                                                                             opt.len_closeness,
                                                                             opt.len_period,
                                                                             opt.len_trend,
                                                                             opt.n_heads,
                                                                             opt.num_layers,
                                                                             opt.skip_dim)
    d_save_path = 'saved_model/separate/{}/SR/{}-{}-{}_{}{}{}_{}_{}_{}/Discriminator'.format(opt.dataset,
                                                                             opt.n_residuals,
                                                                             opt.base_channels,
                                                                             opt.num_epochs,
                                                                             opt.len_closeness,
                                                                             opt.len_period,
                                                                             opt.len_trend,
                                                                             opt.n_heads,
                                                                             opt.num_layers,
                                                                                opt.skip_dim)
    os.makedirs(g_save_path, exist_ok=True)
    os.makedirs(d_save_path, exist_ok=True)
    datapath = os.path.join('data', opt.dataset)
    train_dataloader = get_dataloader_sr(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.n_heads,opt.num_layers,opt.skip_dim,
        opt.scaler_X, opt.scaler_Y, opt.batch_size,'train')  # opt.batch_size=16
    valid_dataloader = get_dataloader_sr(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.n_heads,opt.num_layers,opt.skip_dim,
        opt.scaler_X, opt.scaler_Y, 4, 'valid')

    netG = Generator(scale_factor=UPSCALE_FACTOR, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
                     scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y,ext_flag=True)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))  # param.numel()：返回param中元素的数量
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion = nn.MSELoss()

    if torch.cuda.is_available():
        print("CUDA可用，正在用GPU运行程序")
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    for epoch in range(1, epoch_num+ 1):
        train_bar = tqdm(train_dataloader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        out_path = 'valid_results/epoch{}_{}'.format(epoch, epoch_num + 1)
        # 存储中间预测结果和loss
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for x, ext,target in train_bar:
            # data： batch_size个低分辨率图像
            # target：batch_size个对应高分辨率原图
            batch_size= x.size(0)
            #print("batch size = {}".format(batch_size))
            running_results['batch_sizes'] += batch_size
            #print("running_result = {}".format(running_results))

            ###########################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################

            # print("real_img shape = {}".format(real_img.shape))
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                x = x.cuda()
                ext = ext.cuda()
            fake_img = netG(x, ext)
            # print("fake_img shape = {}".format(fake_img.shape))

            # 网络参数反向传播时，梯度是累积计算的。但其实每个batch间的计算不必累积，因此每个batch要清零
            netD.zero_grad()
            real_out = netD(real_img, ext).mean()
            # print("real_out shape = {} ".format(real_out.shape))
            fake_out = netD(fake_img, ext).mean()
            # print("fake_out shape = {}".format(fake_out.shape))
            d_loss = 1 - real_out + fake_out

            '''
                一般来说每次迭代只需要一次forward和一次backward，即成对出现;
                但有时因为自定义 Loss 的复杂性，需要一次forward()和多个 Loss
                的 backward()来累积同一个网络的grad来更新参数。
                于是，若在当前backward()之后不执行forward()，而是执行另一个backward()
                需要保留计算图，而不是 free 掉。
            '''
            d_loss.backward(retain_graph=True)

            ###########################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_img, real_img)
            g_loss.backward()

            optimizerD.step()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f ' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()

        with torch.no_grad():
            iter = 0
            val_bar = tqdm(valid_dataloader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            for val_x,val_ext, val_hr in val_bar:
                batch_size = val_x.size(0)
                valing_results['batch_sizes'] += batch_size
                x = val_x
                ext = val_ext
                if torch.cuda.is_available():
                    x = x.cuda()
                    ext = ext.cuda()
                    hr = val_hr.cuda()
                sr = netG(x, ext)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f ' % (
                        valing_results['psnr'], valing_results['ssim']))


        # save model parameters
        num = str(epoch / 50)
        torch.save(netG.state_dict(), '{}/final_model_{}.pt'.format(g_save_path, num))
        torch.save(netD.state_dict(), '{}/final_model_{}.pt'.format(d_save_path, num))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

    return


if __name__ == '__main__':
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    train_pre(opt.lr,opt.num_epochs)
    # train_sr(opt.lr,opt.num_epochs)
# -- coding:utf-8 --

import argparse
import os
import numpy as np
from math import log10,sqrt
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data
from torch import nn
from datetime import datetime
from utils.metrics import get_MAE, get_MSE, get_MAPE
import pytorch_ssim
from data_process_pre import get_dataloader_pre
from data_process import get_dataloader
from prediction import TransAm

from sr import Generator,Discriminator


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--dataset', type=str, default='P3', help='which dataset to use')
# 1500 和 100:根据统计数据后粗细粒度每个格子大概的信号容量制定
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--ext_flag', type=bool, default=False, help='whether to use external factor in sr')
#prediction
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=2)
parser.add_argument('--len_trend', type=int, default=0)
parser.add_argument('--external_dim', type=int, default=7)
parser.add_argument('--n_heads', type=int, default=4,
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
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate of prediction in pretrain')
parser.add_argument('--lamda_s', type=float, default=0.1, help='weight of loss between high from prediction_out and high truth')
parser.add_argument('--lamda_p', type=float, default=0.01, help='weight of loss between prediction_out and prediction_truth')
parser.add_argument('--lr_pre', type=float, default=1e-6, help='adam: learning rate of prediction')
parser.add_argument('--lr_sr', type=float, default=1e-4, help='adam: learning rate of super resolution')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--harved_epoch', type=int, default=20, help='halved at every x interval')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
opt = parser.parse_args()
print(opt)

def get_RMSE(pred, real):
    mse = np.mean(np.power(real - pred, 2))
    return sqrt(mse)


def train(lr_pre,lr_sr):
    torch.cuda.manual_seed(opt.seed)
    rmses = [np.inf]
    p_save_path = 'saved_model/to_stage/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/cpt'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    g_save_path = 'saved_model/to_stage/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Generator'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    d_save_path = 'saved_model/to_stage/no_ext(r)/{}/{}_{}/-4-6/{}-{}-{}_{}{}{}_{}_{}/Discriminator'.format(opt.dataset,
                                                                                      opt.lamda_p,
                                                                                      opt.lamda_s,
                                                                                      opt.n_residuals,
                                                                                      opt.base_channels,
                                                                                      opt.num_epochs,
                                                                                      opt.len_closeness,
                                                                                      opt.len_period,
                                                                                      opt.len_trend,
                                                                                      opt.n_heads,
                                                                                      opt.num_layers)
    p_pretrain_path = 'saved_model/separate/{}/cpt/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
                                                               opt.len_closeness,
                                                               opt.len_period,
                                                               opt.len_trend,
                                                               opt.n_heads,
                                                               opt.num_layers,
                                                                opt.skip_dim)
    os.makedirs(p_save_path, exist_ok=True)
    os.makedirs(g_save_path, exist_ok=True)
    os.makedirs(d_save_path, exist_ok=True)
    valid_rmse = torch.zeros((opt.num_epochs,))
    datapath = os.path.join('data', opt.dataset)
    train_dataloader = get_dataloader(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, opt.scaler_Y, opt.batch_size,True,
        'train')  # opt.batch_size=16
    valid_dataloader = get_dataloader(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, opt.scaler_Y, 4, True,'valid')
    netP = TransAm(feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           external_dim=opt.external_dim, dropout=opt.dropout,ext_flag=True)
    netS = Generator(scale_factor=UPSCALE_FACTOR, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
                     scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y, ext_flag=False,residual_flag=False)
    print('# generator parameters:', sum(param.numel() for param in netS.parameters()))  # param.numel()：返回param中元素的数量
    netD = Discriminator(ext_flag=False)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    netP.load_state_dict(torch.load('{}/final_model.pt'.format(p_pretrain_path)))
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        print("CUDA可用，正在用GPU运行程序")
        netP.cuda()
        netS.cuda()
        netD.cuda()
        criterion.cuda()

    optimizerG = optim.Adam([
        {'params': netP.parameters(), 'lr': lr_pre, 'betas': (0.9, 0.999)},
        {'params': netS.parameters(), 'lr': lr_sr, 'betas': (0.9, 0.999)},
    ])
    optimizerD = optim.Adam(netD.parameters(), lr=lr_sr, betas=(opt.b1, opt.b2))
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'p_loss': []}
    min_rmse = 1000
    for epoch in range(1, opt.num_epochs + 1):
        train_bar = tqdm(train_dataloader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netP.train()
        netS.train()
        netD.train()
        out_path = '{}/valid_results/epoch{}_{}'.format(g_save_path,epoch, opt.num_epochs + 1)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for xc, xp, xt, ext,pre, target in train_bar:
            # data： batch_size个低分辨率图像
            # target：batch_size个对应高分辨率原图
            batch_size ,Tc,H,W =  xc.shape
            pre = pre.reshape(batch_size,1,H,W)
            #print("batch size = {}".format(batch_size))
            running_results['batch_sizes'] += batch_size
            #print("running_result = {}".format(running_results))

            ###########################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target
            # print("real_img shape = {}".format(real_img.shape))
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            # print("z shape = {}".format(z.shape))
            if torch.cuda.is_available():
                xc = xc.cuda()
                xp = xp.cuda()
                xt = xt.cuda()
                pre = pre.cuda()
                ext = ext.cuda()

            out_p = netP(xc,xp,xt,ext)
            fake_img= netS(out_p,ext)#预测进入生成器
            loss_pre = criterion(out_p, pre)

            # 网络参数反向传播时，梯度是累积计算的。但其实每个batch间的计算不必累积，因此每个batch要清零

            netD.zero_grad()
            real_out = netD(real_img,ext).mean()
            # print("real_out shape = {} ".format(real_out.shape))
            fake_out = netD(fake_img,ext).mean()

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
            optimizerG.zero_grad()
            g_loss = opt.lamda_s*criterion(fake_img, real_img)+\
                     opt.lamda_p*loss_pre*opt.scaler_X/opt.scaler_Y
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

        netP.eval()
        netS.eval()


        with torch.no_grad():
            iter = 0
            val_bar = tqdm(valid_dataloader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'p_loss':0}

            for val_xc, val_xp,val_xt,val_ext,val_next, val_hr in val_bar:
                batch_size ,Tc,H,W =  val_xc.shape
                val_next = val_next.reshape(batch_size,-1,H,W)
                valing_results['batch_sizes'] += batch_size

                if torch.cuda.is_available():
                    val_xc = val_xc.cuda()
                    val_xp = val_xp.cuda()
                    val_xt = val_xt.cuda()
                    val_next = val_next.cuda()
                    val_hr = val_hr.cuda()
                    val_ext = val_ext.cuda()

                val_out_p = netP(val_xc, val_xp, val_xt, val_ext)
                sr_p = netS(val_out_p,val_ext)

                if iter == 0:
                    pres = val_out_p.reshape(batch_size, H, W)
                else:
                    pres = torch.cat((pres, val_out_p.reshape(batch_size, H, W)), dim=0)
                iter += 1
                batch_ploss = get_MSE(val_out_p.cpu().detach().numpy(), val_next.cpu().numpy())

                batch_mse = ((sr_p - val_hr) ** 2).mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr_p, val_hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (val_hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                valing_results['p_loss'] += batch_ploss * batch_size
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f Loss_p:%.6f' % (
                        valing_results['psnr'], valing_results['ssim'],
                        valing_results['p_loss'] / valing_results['batch_sizes']))

            rmse = sqrt(valing_results['mse'] / len(valid_dataloader.dataset))
            valid_rmse[epoch-1]=rmse
            if rmse < min_rmse:
                min_rmse = rmse
                torch.save(netP.state_dict(), '{}/final_model.pt'.format(p_save_path))
                torch.save(netS.state_dict(), '{}/final_model.pt'.format(g_save_path))
                torch.save(netD.state_dict(), '{}/final_model.pt'.format(d_save_path))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['p_loss'].append(valing_results['p_loss'] / valing_results['batch_sizes'])
        np.save(os.path.join(g_save_path, 'valid_rmse.npy'), valid_rmse.numpy())

    return

if __name__ == '__main__':
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    train(opt.lr_pre,opt.lr_sr)
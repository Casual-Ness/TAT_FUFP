# -- coding:utf-8 --
import os
import numpy as np
import argparse
from utils.metrics import get_MAE, get_MSE, get_MAPE
import torch
from prediction import TransAm
from sr import Generator
from data_process_pre import get_dataloader_pre, get_dataloader_sr, print_model_parm_nums
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
# 1500 和 100:根据统计数据后粗细粒度每个格子大概的信号容量制定
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--dataset', type=str, default='P4',
                    help='which dataset to use')
#prediction
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=3)
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


opt = parser.parse_args()
print(opt)

# test CUDA
cuda = True if torch.cuda.is_available() else False
"""
测试过程
"""
def test_pre(mode):
    model_path = 'saved_model/separate/{}/cpt/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
                                                                     opt.len_closeness,
                                                                     opt.len_period,
                                                                     opt.len_trend,
                                                                     opt.n_heads,
                                                                     opt.num_layers,
                                                                    opt.skip_dim)
    model = TransAm(feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                    skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                    len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                    external_dim=opt.external_dim, dropout=opt.dropout).to(device)  # 得到网络实例模型
    model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path),map_location=torch.device('cpu')))
    model.eval()
    save_path = os.path.join('data/{}'.format(opt.dataset), mode)
    os.makedirs(save_path, exist_ok=True)

    # 初始化记忆单元,shape是(batch,num_layer,hidden_len)

    MSE = 100000
    datapath = os.path.join('data', opt.dataset)
    dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X, 1,True, mode=mode)
    total_mse, total_mae, total_mape, total_rmse = 0, 0, 0, 0

    for n, (xc, xp, xt, ext, lable) in enumerate(dataloader):
        los = 0
        l = lable.to(device)
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        ext = ext.to(device)
        B, Tc, H, W = xc.shape
        with torch.no_grad():
            out = model(xc, xp, xt, ext)
            # 计算和预期输出之间的MSE损失
        pre = out.cpu().detach().numpy() * opt.scaler_X
        l = l.cpu().detach().reshape(B,1, H, W).numpy() * opt.scaler_X
        total_mse += get_MSE(pre, l)
        total_mae += get_MAE(pre, l)
        total_mape += get_MAPE(pre, l)


    mse = total_mse / len(dataloader.dataset)
    mae = total_mae / len(dataloader.dataset)
    mape = total_mape / len(dataloader.dataset)
    rmse = np.sqrt(mse)

    f = open('{}/results.txt'.format(model_path), 'a')
    f.write("{}:\tRMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\n".format(mode,rmse, mae, mape))
    f.close()

    print('Test MSE = {:.6f} ,MAE = {:.6f}, MAPE = {:.6f},RMSE = {:.6f}'.format(mse, mae, mape, rmse))


def test_sr(mode = 'test'):
    model_path = 'saved_model/separate/{}/SR/{}-{}-{}_{}{}{}_{}_{}_{}/Generator'.format(opt.dataset,
                                                                             opt.n_residuals,
                                                                             opt.base_channels,
                                                                             opt.num_epochs,
                                                                             opt.len_closeness,
                                                                             opt.len_period,
                                                                             opt.len_trend,
                                                                             opt.n_heads,
                                                                             opt.num_layers,
                                                                            opt.skip_dim)
    model = Generator(scale_factor=opt.upscale_factor, n_residual_block=opt.n_residuals, base_channel=opt.base_channels,
                     scaler_x=opt.scaler_X, scaler_y=opt.scaler_Y,ext_flag=True)
    # load test set
    datapath = os.path.join('data', opt.dataset)
    dataloader = get_dataloader_sr(datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.n_heads,opt.num_layers,
        opt.skip_dim,opt.scaler_X, opt.scaler_Y, opt.batch_size,mode)
    save_path = 'test_result/separate/{}/SR/{}-{}-{}_{}{}{}_{}_{}_{}'.format(opt.dataset,
                                                                             opt.n_residuals,
                                                                             opt.base_channels,
                                                                             opt.num_epochs,
                                                                             opt.len_closeness,
                                                                             opt.len_period,
                                                                             opt.len_trend,
                                                                             opt.n_heads,
                                                                             opt.num_layers,
                                                                             opt.skip_dim)  # result save path
    os.makedirs(save_path, exist_ok=True)
    min_rmse = 1000
    min_mae = 1000
    min_mape = 100
    k = 0
    for i in range(100, 300 + 1):
        num = str(i / 50)
        model.load_state_dict(
            torch.load('{}/final_model_{}.pt'.format(model_path, num)))
        model.eval()
        if cuda:
            model.cuda()
        np.set_printoptions(suppress=True)

        total_mse, total_mae, total_mape = 0, 0, 0

        for j, (x, ext, test_labels) in enumerate(dataloader):
            if cuda:
                x = x.cuda()
                ext = ext.cuda()
                test_labels = test_labels.cuda()
            hr = model(x, ext)
            hr = hr.cpu().detach().numpy() * opt.scaler_Y
            test_labels = test_labels.cpu().detach().numpy() * opt.scaler_Y

            total_mse += get_MSE(hr, test_labels) * len(x)
            total_mae += get_MAE(hr, test_labels) * len(x)
            total_mape += get_MAPE(hr, test_labels) * len(x)


        rmse = np.sqrt(total_mse / len(dataloader.dataset))
        mae = total_mae / len(dataloader.dataset)
        mape = total_mape / len(dataloader.dataset)
        if rmse<min_rmse:
            k = i
            min_rmse = rmse
            min_mae = mae
            min_mape = mape

        f = open('{}/results.txt'.format(save_path), 'a')
        f.write("i/301\t{}\tnum\t{}\tRMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\n".format(i, num, rmse, mae, mape))
        f.close()

        print('{:d}: Test RMSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}'.format(i, rmse, mae, mape))

    f = open('{}/results.txt'.format(save_path), 'a')
    f.write("{}\ti={}\tmin_RMSE={:.6f}\tmin_MAE={:.6f}\tmin_MAPE={:.6f}\n".format(mode,k, min_rmse, min_mae, min_mape))
    f.close()
    print('{}\ti={}\tmin_RMSE={:.6f}\tmin_MAE={:.6f}\tmin_MAPE={:.6f}\n'.format(mode,k, min_rmse, min_mae, min_mape))
    return


if __name__ == '__main__':
    # test_pre('train')
    # test_pre('valid')
    test_pre('test')
    # test_sr('test')

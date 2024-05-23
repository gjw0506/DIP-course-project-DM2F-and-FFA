# coding: utf-8
import argparse
import os
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from FFA_new import FFA
from model import DM2FNet
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset
from tools.utils import AvgMeter, check_mkdir
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tools.utils import AvgMeter, check_mkdir, sliding_forward
# DM2FNet_woPhy和DM2FNet的区别在于知不知道a和t
def parse_args():
    parser = argparse.ArgumentParser(description='Train a FFA')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 300000,
    'train_batch_size': 2,
    'last_iter': 110000,
    'lr': 1e-4,
    # 'lr_decay': 0.9,
    # 'weight_decay': 0,
    # 'momentum': 0.9,
    'snapshot': 'iter_110000_loss_0.03838',
    'val_freq': 10000,
    'crop_size': 120
}



def main():
    net = FFA(gps=3, blocks=19).cuda().train()
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=cfgs['lr'], betas = (0.9, 0.999), eps=1e-08)
    # net = nn.DataParallel(net)
    # # 对偏置项和非偏置项（通常是权重）设置不同的学习率（lr）
    # optimizer = optim.Adam([
    #     {'params': [param for name, param in net.named_parameters()
    #                 if name[-4:] == 'bias' and param.requires_grad],
    #      'lr': 2 * cfgs['lr']},
    #     {'params': [param for name, param in net.named_parameters()
    #                 if name[-4:] != 'bias' and param.requires_grad],
    #      'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    # ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        # optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
        #                                                   args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        # optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        # optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer)


def train(net, optimizer):
    curr_iter = cfgs['last_iter']

    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()

        for data in train_loader:
            # optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
            #                                   ** cfgs['lr_decay']
            # optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
            #                                   ** cfgs['lr_decay']

            haze, _, _, gt, _ = data

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            pred = net(haze)
            loss = criterion(pred,gt)

            loss.backward()

            optimizer.step()

            # update recorder
            train_loss_record.update(loss.item(), batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f]' % \
                  (curr_iter, train_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter)#, optimizer)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter):#, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    mse_record, psnr_record, ssim_record = AvgMeter(), AvgMeter(), AvgMeter()
    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = sliding_forward(net, haze, crop_size=cfgs['crop_size'])

            loss = criterion(dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))
            
            for i in range(len(haze)):
                r = dehaze[i].cpu().numpy().transpose([1, 2, 0])  # data range [0, 1]
                g = gt[i].cpu().numpy().transpose([1, 2, 0])
                mse = mean_squared_error(g, r)
                psnr = peak_signal_noise_ratio(g, r)
                ssim = structural_similarity(g, r, data_range=1, multichannel=True,
                                             gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                # ciede = calculate_ciede2000(g, r)
                mse_record.update(mse)
                psnr_record.update(psnr)
                ssim_record.update(ssim)
                # ciede_record.update(ciede)

    log = '[validate]: [iter {}], [loss {:.5f}] [PSNR {:.4f}] [SSIM {:.4f}][MSE {:.4f}]'.format(
        curr_iter + 1, loss_record.avg, psnr_record.avg, ssim_record.avg, mse_record.avg)#, ciede_record.avg)

    snapshot_name = 'iter_%d_loss_%.5f' % (curr_iter + 1, loss_record.avg)
    print(log)
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    # torch.save(optimizer.state_dict(),
    #            os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=4)

    criterion = nn.L1Loss().cuda()
    log_path = os.path.join(args.ckpt_path, args.exp_name, '1.txt')

    main()

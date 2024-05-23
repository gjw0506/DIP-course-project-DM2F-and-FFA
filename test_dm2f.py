# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from skimage.color import rgb2lab, deltaE_ciede2000
from tools.config import OHAZE_ROOT, TEST_SOTS_ROOT, TEST_HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward, sliding_forward_for_hazerd
from model import DM2FNet, DM2FNet_woPhy
from datasets import OHazeDataset , SotsDataset, hazerdDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
# exp_name = 'O-Haze'
exp_name = 'Hazerd'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
    'snapshot': 'iter_40000_loss_0.01395_lr_0.000000'
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    'HAZERD': TEST_HAZERD_ROOT,
    # 'O-Haze': OHAZE_ROOT,
}

to_pil = transforms.ToPILImage()

def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HAZERD' in name:
                net = DM2FNet().cuda()
                dataset = hazerdDataset(root)
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, mses, ciedes = [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                elif 'HAZERD' in name:
                    res = sliding_forward_for_hazerd(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                for i in range(len(fs)): # 遍历的意义？fs是1
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                    ssims.append(ssim)
                    mse = mean_squared_error(gt, r)
                    mses.append(mse)
                    ciede = np.mean(deltaE_ciede2000(rgb2lab(gt), rgb2lab(r)))
                    ciedes.append(ciede)

                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, MSE {:.4f}, CIEDE{:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, mse, ciede))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, MSE: {np.mean(mses):.6f}, CIEDE2000: {np.mean(ciedes):.6f}")


if __name__ == '__main__':

    main()

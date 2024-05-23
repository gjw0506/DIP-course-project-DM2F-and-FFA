import os
import os.path

import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import ToTensor
to_tensor = ToTensor()
# c,h,w[0,1]



def make_dataset_its(root):
    items = []
    for img_name in os.listdir(os.path.join(root, 'hazy')):
        idx0, idx1, ato = os.path.splitext(img_name)[0].split('_')
        gt = os.path.join(root, 'clear', idx0 + '.png')
        trans = os.path.join(root, 'trans', idx0 + '_' + idx1 + '.png')
        haze = os.path.join(root, 'hazy', img_name)
        items.append([haze, trans, float(ato), gt])

    return items



def make_dataset_ohaze(root: str, mode: str):
    img_list = []
    for img_name in os.listdir(os.path.join(root, mode, 'hazy')):
        gt_name = img_name.replace('hazy', 'GT')
        assert os.path.exists(os.path.join(root, mode, 'gt', gt_name))
        img_list.append([os.path.join(root, mode, 'hazy', img_name),
                         os.path.join(root, mode, 'gt', gt_name)])
    return img_list




def random_crop(size, haze, gt, extra=None):
    w, h = haze.size
    assert haze.size == gt.size

    if w < size or h < size:
        haze = transforms.Resize(size)(haze)
        gt = transforms.Resize(size)(gt)
        w, h = haze.size

    x1 = random.randint(0, w - size)
    y1 = random.randint(0, h - size)

    _haze = haze.crop((x1, y1, x1 + size, y1 + size))
    _gt = gt.crop((x1, y1, x1 + size, y1 + size))

    if extra is None:
        return _haze, _gt
    else:
        # extra: trans or predict
        assert haze.size == extra.size
        _extra = extra.crop((x1, y1, x1 + size, y1 + size))
        return _haze, _gt, _extra



class ItsDataset(data.Dataset):
    """
    For RESIDE Indoor
    """

    def __init__(self, root, flip=False, crop=None):
        self.root = root
        self.imgs = make_dataset_its(root)
        self.flip = flip
        self.crop = crop

    def __getitem__(self, index):
        haze_path, trans_path, ato, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        trans = Image.open(trans_path).convert('L')
        gt = Image.open(gt_path).convert('RGB')

        assert haze.size == trans.size
        assert trans.size == gt.size

        if self.crop:
            haze, gt, trans = random_crop(self.crop, haze, gt, trans)

        if self.flip and random.random() < 0.5:
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            trans = trans.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        haze = to_tensor(haze)
        trans = to_tensor(trans)
        gt = to_tensor(gt)
        gt_ato = torch.Tensor([ato]).float()

        return haze, trans, gt_ato, gt, name

    def __len__(self):
        return len(self.imgs)





def make_dataset(root):
    return [(os.path.join(root, 'hazy', img_name),
             os.path.join(root, 'gt', img_name))
            for img_name in os.listdir(os.path.join(root, 'hazy'))]


class SotsDataset(data.Dataset):
    def __init__(self, root, mode=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.mode = mode

    def __getitem__(self, index):
        haze_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        haze = to_tensor(haze)

        idx0 = name.split('_')[0]
        gt = Image.open(os.path.join(self.root, 'gt', idx0 + '.png')).convert('RGB')
        gt = to_tensor(gt)
        if gt.shape != haze.shape:
            # crop the indoor images
            print(f"{name}")
            gt = gt[:, 10: 470, 10: 630]

        return haze, gt, name

    def __len__(self):
        return len(self.imgs)


class OHazeDataset(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.imgs = make_dataset_ohaze(root, mode)

    def __getitem__(self, index):
        haze_path, gt_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        img = Image.open(haze_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        if 'train' in self.mode:
            # img, gt = random_crop(416, img, gt)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

            rotate_degree = np.random.choice([-90, 0, 90, 180])
            img, gt = img.rotate(rotate_degree, Image.BILINEAR), gt.rotate(rotate_degree, Image.BILINEAR)

        return to_tensor(img), to_tensor(gt), name#[0,1]

    def __len__(self):
        return len(self.imgs)

def make_dataset_hazerd(root):
    return [os.path.join(root, 'simu', img_name)
            for img_name in os.listdir(os.path.join(root, 'simu')) if img_name.endswith('.jpg')]


class hazerdDataset(data.Dataset):
    def __init__(self, root, mode=None):
        self.root = root
        self.imgs = make_dataset_hazerd(root)
        self.mode = mode

    def __getitem__(self, index):
        haze_path = self.imgs[index]
        name = os.path.splitext(os.path.split(haze_path)[1])[0]

        haze = Image.open(haze_path).convert('RGB')
        haze = to_tensor(haze)

        idx0 = name.split('_')[1]
        gt = Image.open(os.path.join(self.root, 'img', 'IMG_' + idx0 + '_RGB.jpg')).convert('RGB')
        gt = to_tensor(gt)
        assert(gt.shape == haze.shape)

        return haze, gt, name

    def __len__(self):
        return len(self.imgs)
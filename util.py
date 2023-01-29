from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os
from os.path import exists, join
from glob import glob
from PIL import Image
from thermal_utils import processThermalImage
from random import sample


def get_mean_std(dataset_folder):

    r_mean, g_mean, b_mean = 0, 0, 0
    r_mean_sq, g_mean_sq, b_mean_sq = 0, 0, 0
    n = 0

    from tqdm import tqdm
    for file_name in tqdm(os.listdir(dataset_folder)):
        
        if not file_name.endswith(".png") or file_name.endswith(".jpg"):
            continue
        
        image_array = np.array(path_to_pil_thermal_img(os.path.join(dataset_folder, file_name)), dtype=float)/255
        r_channel, g_channel, b_channel = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        r_channel_sq, g_channel_sq, b_channel_sq = (r_channel ** 2), (g_channel ** 2), (b_channel ** 2)

        n += 1
        r_mean += (np.mean(r_channel) - r_mean) / n
        g_mean += (np.mean(g_channel) - g_mean) / n
        b_mean += (np.mean(b_channel) - b_mean) / n
        r_mean_sq += (np.mean(r_channel_sq) - r_mean_sq) / n
        g_mean_sq += (np.mean(g_channel_sq) - g_mean_sq) / n
        b_mean_sq += (np.mean(b_channel_sq) - b_mean_sq) / n
    
    mean = (r_mean, g_mean, b_mean)
    std = (math.sqrt(r_mean_sq - r_mean**2), math.sqrt(g_mean_sq - g_mean**2), math.sqrt(b_mean_sq - b_mean**2))
    print('mean: ', mean)
    print('std: ', std)
    return tuple([round(x, 4) for x in mean]), tuple([round(x, 4) for x in std])

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def path_to_pil_thermal_img(path):
    image = processThermalImage(path, crop=True, undistort=True, 
                            normalize_minmax=True, keeptype=True)
    return Image.fromarray(image).convert("RGB")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, labels=None, transform=None):
        super().__init__()
        self.root = root
        self.labels = labels
        self.transform = transform

        if not exists(root):
            raise FileNotFoundError(f"Folder {root} does not exist")
        
        self.data_path = []
        for folder in os.listdir(root):
            data_path = join(root, folder, 'image', 'stereo_thermal_14_left')
            self.data_path.extend(sorted(glob(join(data_path, "*.png"), recursive=True)))

        # self.data_path = sorted(glob(join(root, "**", "*.png"), recursive=True))
        print(f"Found {len(self.data_path)} images in {root} folder")
        self.data_path = sample(self.data_path, 10000) if len(self.data_path)>10000 else self.data_path
        print(f"After random sampling -> {len(self.data_path)} images")


    def __getitem__(self, index):
        img = path_to_pil_thermal_img(self.data_path[index])
        # label = self.labels[index]
        label = 0

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_path)



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

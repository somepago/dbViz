'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary
from options import options
from utils import simple_lapsed_time
from utils import produce_plot_alt

#args.plot_path
parser = argparse.ArgumentParser(description='Argparser for sanity check')

parser.add_argument('--net', default='ResNet', type=str)
parser.add_argument('--plot_path', type=str, default=None)
parser.add_argument('--baseset', default='CIFAR10', type=str,
                            choices=['CIFAR10', 'CIFAR100','SVHN',
                            'CIFAR100_label_noise'])
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--imgs', default=None,
                        type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--temp', default=1.0, type=float)
parser.add_argument('--plot_method', default='greys', type=str)
parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--adv', action='store_true', help='Adversarially attack images?')

args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = get_data(args)

def plot(net_name, load_path, plot_path):
    print('###############################')
    print(net_name)
    print(load_path)
    print(plot_path)
    start = time.time()
    args.net = net_name
    net = get_model(args, device)
    if torch.cuda.device_count() > 1:
        net.module.load_state_dict(torch.load(load_path))
    else:
        net.load_state_dict(torch.load(load_path))

    # test_acc, predicted = test(args, net, testloader, device)
    # print(test_acc)
    end = time.time()
    simple_lapsed_time("Time taken to train/load the model", end - start)


    start = time.time()
    if args.imgs is None:
        # images, labels = get_random_images(trainloader.dataset)
        images, labels = get_random_images(testloader.dataset)
    elif -1 in args.imgs:
        dummy_imgs, _ = get_random_images(testloader.dataset)
        images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
    elif -10 in args.imgs:
        image_ids = args.imgs[0]
        #         import ipdb; ipdb.set_trace()
        images = [testloader.dataset[image_ids][0]]
        labels = [testloader.dataset[image_ids][1]]
        for i in list(range(2)):
            temp = torch.zeros_like(images[0])
            if i == 0:
                temp[0, 0, 0] = 1
            else:
                temp[0, -1, -1] = 1

            images.append(temp)
            labels.append(0)

    #         dummy_imgs, _ = get_random_images(testloader.dataset)
    #         images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
    # incomplete
    else:
        image_ids = args.imgs
        images = [testloader.dataset[i][0] for i in image_ids]
        labels = [testloader.dataset[i][1] for i in image_ids]
        print(labels)

    if args.adv:
        adv_net = AttackPGD(net, trainloader.dataset)
        adv_preds, imgs = adv_net(torch.stack(images).to(device), torch.tensor(labels).to(device))
        images = [img.cpu() for img in imgs]

    planeloader = make_planeloader(images, args)
    preds = decision_boundary(args, net, planeloader, device)

    sampl_path = '_'.join(list(map(str, args.imgs)))
    args.plot_path = plot_path
    plot = produce_plot_alt(args.plot_path, preds, planeloader, images, labels, trainloader, temp=args.temp)

    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end - start)


Archs = ['ResNet', 'VGG' , 'GoogLeNet' , 'DenseNet' , 'MobileNet']
all_models_path = './saved_models/'
all_final_plot_path = './saved_final_imgs'

for arch in Archs:
    #net_name, load_path, plot_path
    print('########################################################################')
    net_name = arch

    for originals in ['naive', 'mixup', 'cutmix']:
        load_path = all_models_path + originals + '/' + arch + '_cifar10.pth'
        plot_path = all_final_plot_path + '/soft_distillation/' + arch + '/' + originals
        plot(net_name, load_path, plot_path)


    for from_arch in Archs:
        load_path = all_models_path + '/soft_distillation/from_' + from_arch + '/' + arch + '_cifar10.pth'
        plot_path = all_final_plot_path  + '/soft_distillation/' + arch + '/' + 'from_' + from_arch
        plot(net_name, load_path, plot_path)

    for from_method in ['cutmix', 'mixup']:
        load_path = all_models_path + '/soft_distillation/from_' + from_method + '/' + arch + '_cifar10.pth'
        plot_path = all_final_plot_path + '/soft_distillation/' + arch + '/' + 'from_' + from_method
        plot(net_name, load_path, plot_path)


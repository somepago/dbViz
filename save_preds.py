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
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)
ious = []

net_dict = dict([('vit_pt','ViT_pt'), ('resnet','ResNet'), ('densenet','DenseNet'), ('vgg','VGG'), ('fcnet','FCNet'), ('wideresnet', 'WideResNet')])

def get_net_name(path):
    #temp_name = path.split('/')[-1].split('.')[0]
    temp_name = path.split('/')[-1].split('_naive')[0]
    return net_dict[temp_name.lower()]


os.makedirs(args.load_net + '/predictions', exist_ok=True)
paths = [os.path.join(args.load_net, p) for p in os.listdir(args.load_net) if 'pred' not in p]

for path in paths:
    print(path)
    if 'ViT' not in path:
        if 'wide' in  path:
            args.net = 'WideResNet'
            #args.widen_factor = int(path.split('/')[-1].split('.')[0].split('_')[1])
            args.widen_factor = 10
        else:
            args.net = get_net_name(path)
        net = get_model(args, device)
        net.load_state_dict(torch.load(path))
        pred_arr = []
        for run in range(args.epochs):
            random.seed(a=(args.set_data_seed+run), version=2)
            images, labels, image_ids = get_random_images(testloader.dataset)
            planeloader = make_planeloader(images, args)
            preds = decision_boundary(args, net, planeloader, device)
            pred_arr.append(torch.stack(preds).argmax(1).cpu())
        torch.save(pred_arr, args.load_net + '/predictions/' + path.split('/')[-1].split('.pth')[0] + 'preds.pth')
        #torch.save(pred_arr, args.save_net + '/predictions/' + path.split('/')[-1].split('.pth')[0] + 'preds.pth')
        #torch.save(pred_arr, os.path.join(args.load_net, f"{args.net}_{path.split('/')[-1].split('seed')[1]}"))
            

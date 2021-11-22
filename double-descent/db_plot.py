'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
# from utils import progress_bar



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=200, type=int, help='total number of training epochs')      
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=64, type=int, help='depth of model')      
parser.add_argument('--active_log', action='store_true') 
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--noise_rate', default=0.2, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--model_path', type=str, default='./checkpoint')
parser.add_argument('--mixup', action='store_true') 


parser.add_argument('--plot_animation', action='store_true') 
parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--range_l', default=0.5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=0.5, type=float, help='how far `right` to go in the plot')
parser.add_argument('--temp', default=5.0, type=float)
parser.add_argument('--plot_method', default='train', type=str)
parser.add_argument('--plot_train_class', default=9, type=int)

parser.add_argument('--plot_train_imgs', action='store_true')
parser.add_argument('--plot_path', type=str, default='./imgs')
parser.add_argument('--extra_path', type=str, default=None)
parser.add_argument('--imgs', default=None,
                            type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')

parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
# parser.add_argument('--run_name', type=str, default='temp')    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

if args.mixup:
    args.model_path = './checkpoint/mixup'

# best_acc = 0  # best test accuracy
# best_epoch = 0
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# if not os.path.isdir('./checkpoint'):
#     os.mkdir('./checkpoint')
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

torch.manual_seed(args.set_data_seed)

from data_noisy import cifar10Nosiy
trainset_noisy = cifar10Nosiy(root='./data', train=True,transform=transform_train, download=True,
                                            asym=args.asym,
                                            nosiy_rate=0.2)

train_class = args.plot_train_class
l1 = np.where(np.array(trainset_noisy.true_targets) == train_class)[0]
l2 = np.where(np.array(trainset_noisy.targets) != np.array(trainset_noisy.true_targets))[0]
if args.noise_rate > 0: 
    trainset = cifar10Nosiy(root='./data', train=True,transform=transform_train, download=True,
                                            asym=args.asym,
                                            nosiy_rate=args.noise_rate)
    
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if args.plot_method == 'test':
    image_ids = args.imgs
    images = [testloader.dataset[i][0] for i in image_ids]
    labels = [testloader.dataset[i][1] for i in image_ids]
else:
    if args.plot_method == 'train_ids':
        image_ids = args.imgs
    else:
        img_locs = args.imgs
        if args.plot_method == '3corr':
            image_ids = np.setdiff1d(l1,l2)[img_locs]
        elif args.plot_method == 'train_1inc':
            inc_id = np.intersect1d(l1,l2)[img_locs[0]]
            corr_id = np.setdiff1d(l1,l2)[img_locs[1:]]
            image_ids = [corr_id[0],corr_id[1],inc_id]
        elif args.plot_method == 'train_2inc':
            inc_id = np.intersect1d(l1,l2)[img_locs[0:1]]
            corr_id = np.setdiff1d(l1,l2)[img_locs[2]]
            image_ids = [corr_id[0],inc_id[0],inc_id[1]]
        else:
            raise'Case not written'

    images = [trainloader.dataset[i][0] for i in image_ids]
    labels = [trainloader.dataset[i][1] for i in image_ids]

if args.plot_method != 'test' and args.noise_rate > 0:
    true_labels = [trainset.true_targets[i] for i in image_ids]
    if np.sum(np.array(labels) == np.array(true_labels)) == 3:
        true_labels = None
else:
    true_labels = None
print(image_ids, labels,true_labels)
# import ipdb; ipdb.set_trace()
sampleids = '_'.join(list(map(str,image_ids)))

mp = args.model_path
modelpath = f'{mp}/{args.set_seed}/{args.set_data_seed}/dd_Adam_{args.noise_rate}noise_{args.k}k/ckpt.pth'
net = make_resnet18k(args.k)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['net'])
net.eval()

from db.data import make_planeloader
from db.evaluation import decision_boundary
from db.utils import produce_plot_alt,produce_plot_x,connected_components,produce_plot_sepleg

planeloader = make_planeloader(images, args)
preds = decision_boundary(args, net, planeloader, device)
plot_path = f'{args.plot_path}_{sampleids}_{args.noise_rate}_rn{args.k:02d}' 
if args.noise_rate == 0:
    title = f"k = {args.k}"
else:
    title = f"k = {args.k}, noise = {args.noise_rate}"
# plot = produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader, temp=args.temp)
plot = produce_plot_sepleg(plot_path, preds, planeloader, images, labels, trainloader, title = title, temp=args.temp,true_labels = true_labels)
# ncc = connected_components(preds,args,plot_path)
# print(ncc)
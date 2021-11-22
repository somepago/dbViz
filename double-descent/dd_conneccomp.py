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



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--k', default=64, type=int, help='depth of model')      
parser.add_argument('--noise_rate', default=0.2, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')

parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--range_l', default=0.5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=0.5, type=float, help='how far `right` to go in the plot')
# parser.add_argument('--temp', default=5.0, type=float)
parser.add_argument('--plot_method', default='train', type=str)
parser.add_argument('--mixup', action='store_true') 

# parser.add_argument('--plot_train_class', default=4, type=int)

# parser.add_argument('--plot_train_imgs', action='store_true')
# parser.add_argument('--plot_path', type=str, default='./imgs')
# parser.add_argument('--extra_path', type=str, default=None)
# parser.add_argument('--imgs', default=None,
#                             type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')



parser.add_argument('--active_log', action='store_true') 

parser.add_argument('--num_samples', default=100 , type=int)
parser.add_argument('--num_iters', default=10 , type=int)
# parser.add_argument('--delta', default=0.5, type=float, help='how far from image')

parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
# parser.add_argument('--run_name', type=str, default='temp')    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args.eval = True

if args.mixup:
    args.model_path = './checkpoint/mixup'
else:
    args.model_path = './checkpoint'
    
if args.active_log:
    import wandb
    if not args.mixup: 
        wandb.init(project="dd_connect_comp", name = f'frag_{args.plot_method}_{args.noise_rate}noise_{args.k}k')
    else:
        wandb.init(project="dd_connect_comp", name = f'frag_{args.plot_method}_{args.noise_rate}noise_{args.k}k_wmixup')
    wandb.config.update(args)



best_acc = 0  # best test accuracy
best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if not os.path.isdir('./checkpoint'):
    os.mkdir('./checkpoint')
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
if args.noise_rate > 0: 
    from data_noisy import cifar10Nosiy
    trainset = cifar10Nosiy(root='./data', train=True,transform=transform_train, download=True,
                                            asym=args.asym,
                                            nosiy_rate=args.noise_rate)
    
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


print('RAN FINE TILL HERE!')

if args.noise_rate > 0:
    l_mis = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0] #mislabeled images
    l_corr = np.where(np.array(trainset.targets) == np.array(trainset.true_targets))[0] #correctly labeled images
l_all = np.arange(len(trainset.targets))

from db.data import make_planeloader
from db.evaluation import decision_boundary
from db.utils import connected_components


def num_connected_components(dlist1,loader, num_samples,net,device,args):
    cc_list = []
    for i in range(num_samples):
        dirlist = np.random.choice(dlist1, 3)
        images = [loader.dataset[i][0] for i in dirlist]
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        ncc = connected_components(preds,args)
        cc_list.append(ncc)

        if i%100==0:
            if args.active_log:
                wandb.log({'iteration':i})
    return cc_list

    
mp = args.model_path
modelpath = f'{mp}/dd_Adam_{args.noise_rate}noise_{args.k}k/ckpt.pth'
net = make_resnet18k(args.k)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['net'])
net.eval()

connected_comp_count = {
    'aa' : num_connected_components(l_all,trainloader,args.num_samples,net,device,args)
} 
if args.active_log:
                wandb.log({'aa': np.mean(connected_comp_count['aa'])})
                wandb.log({'aa_median': np.median(connected_comp_count['aa'])})

# import ipdb; ipdb.set_trace()
# savepath = f'ddquant/connectedcomp/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl'
# import json
# with open(f"{savepath}.txt", 'w') as f: 
#     f.write(str(connected_comp_count)

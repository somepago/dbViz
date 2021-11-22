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
from models import *
import pandas as pd

from db.data import make_planeloader
from db.evaluation import decision_boundary
from db.utils import get_random_images, simple_lapsed_time
import wandb
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--k', default=64, type=int, help='depth of model')      
parser.add_argument('--noise_rate', default=0.2, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')

parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--range_l', default=0.5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=0.5, type=float, help='how far `right` to go in the plot')
parser.add_argument('--plot_method', default='train', type=str)
parser.add_argument('--mixup', action='store_true') 
parser.add_argument('--active_log', action='store_true') 
parser.add_argument('--num_samples', default=100 , type=int)
parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
parser.add_argument('--paths', default=None,
                            type=lambda s: [item for item in s.split(';')], help='which images ids to plot')

args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project="dd_iou", name = f'iou_{args.paths}')
wandb.config.update(args)

def get_which_loader(nr,args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    torch.manual_seed(args.set_data_seed)
    if args.plot_method =='test':
        testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)
        return testloader
    else:
        if nr > 0: 
            from data_noisy import cifar10Nosiy
            trainset = cifar10Nosiy(root='./data', train=True,transform=transform_train, download=True,
                                                    asym=args.asym,
                                                    nosiy_rate=nr)  
        else:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        return trainloader



# torch.manual_seed(args.set_seed)

def calculate_iou_simple(pred_arr1, pred_arr2):
    #pred_arr1 = torch.stack(pred_arr1).argmax(1)
    #pred_arr2 = torch.stack(pred_arr2).argmax(1)
    diff = pred_arr1.shape[0] - (pred_arr1 - pred_arr2).count_nonzero()
    iou = diff / pred_arr1.shape[0]
    return iou.cpu()

    
paths = args.paths


iou_mat = torch.zeros((len(paths), len(paths)))
iou_count = torch.zeros((len(paths), len(paths)))
# path_pairs = get_path_pairs(paths)

# print(f'Calculating IOU for {sum([len(path) for path in paths])} networks')
predictions = {}
start = time.time()
for i, group in enumerate(paths):
    start = time.time()
    # import ipdb; ipdb.set_trace()

    k,sd,dsetseed,nr = eval(group)
    loader = get_which_loader(nr,args)
    modelpath = f'./checkpoint/{sd}/{dsetseed}/dd_Adam_{nr}noise_{k}k/ckpt.pth'
    net = make_resnet18k(k)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    pred_arr = []
    wandb.log({'Inmodel':i})
    for run in range(args.num_samples):
        random.seed(a=(args.set_data_seed+run), version=2)
        images, labels, image_ids = get_random_images(loader.dataset)
        # print(image_ids,args.set_data_seed+run)
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        pred_arr.append(torch.stack(preds).argmax(1).cpu())
        wandb.log({'iteration':run})
    predictions[group] = torch.cat(pred_arr)
print(f'Done calculating predictions for {len(paths)} networks')
# end = time.time()
# simple_lapsed_time(f"Average time for {sum([len(path) for path in paths])} networks, and {args.num_samples} runs :", (end - start) / sum([len(path) for path in paths])/args.epochs)
# simple_lapsed_time(f'Total time:', (end - start))
           

from itertools import combinations
allcombs = combinations(paths, 2) 

iou_df = wandb.Table(columns=['model1','model2','IOU'])

from pathlib import Path
my_file = Path('./dd_iou_paper.csv')
if not my_file.is_file():
    iou_df_pd =  pd.DataFrame(columns=['model1','model2','IOU','resolution','num_samples','plot_method'])
    iou_df_pd.to_csv(my_file,index = None)
else:
    iou_df_pd = pd.read_csv(my_file)

for t1, t2 in allcombs:
    iou = calculate_iou_simple(predictions[t1].to(device), predictions[t2].to(device)).item()
    iou_dict = {'model1':t1,'model2':t2,'IOU':iou,'resolution':args.resolution,'num_samples':args.num_samples,'plot_method':args.plot_method}
    iou_df_pd = iou_df_pd.append(iou_dict,ignore_index=True)
    iou_df.add_data(t1,t2,iou)
# import ipdb; ipdb.set_trace()
wandb.log({"table_key": iou_df})

iou_df_pd.to_csv(my_file,index = None)

# start = time.time()
# total = 0
# running_total = 0
# for i in range(len(predictions)):
#     for j in range(i, len(predictions)):
#         if i == j:
#             for k in range(len(predictions[i])):
#                 for l in range(k+1, len(predictions[j])):
                    
#                     iou_mat[i, j] += iou
#                     iou_count[i, j] += 1
#                     total += 1
#                     end = time.time()
#                     running_total += end-start
#                     running_avg = running_total / total
#                     if total % 10 == 0:
#                         print(iou_mat / iou_count)
#                         # simple_lapsed_time("Estimated time remaining:", running_avg * len(path_pairs) - running_total)
                        
#         else:
#             for k in range(len(predictions[i])):
#                 for l in range(len(predictions[j])):
#                     iou = calculate_iou_simple(predictions[i][k].to(device), predictions[j][l].to(device))
#                     iou_mat[i, j] += iou
#                     iou_count[i, j] += 1
#                     total += 1
#                     end = time.time()
#                     running_total += end-start
#                     running_avg = running_total / total
#                     if total % 10 == 0:
#                         print(iou_mat / iou_count)
#                         # simple_lapsed_time("Estimated time remaining:", running_avg * len(path_pairs) - running_total)

# print(iou_mat / iou_count)
                    
            

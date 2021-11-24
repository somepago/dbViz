'''Train CIFAR10 with PyTorch.'''
import torch
import random

import os
import argparse

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import decision_boundary
from options import options
from utils import simple_lapsed_time

'''
This module calculates and saves prediction arrays for different saved models. 
E.g. if you have saved models with the structure: 
/path/to/saved/models
    - model_1.pth
    - model_2.pth
    - model_3.pth

then this script will make a new folder /path/to/saved/models/predictions which
will save the following prediction arrays:
/path/to/save/models/predictions
    - model_1_preds.pth
    - model_2_preds.pth
    - model_3_preds.pth

Note: the original model paths should be of the form: 
ResNet18.pth
...
i.e. the name of the model used should be the same as the model in model.py  
'''

args = options().parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)

paths = [os.path.join(args.load_net, p) for p in os.listdir(args.load_net) if 'pred' not in p]

for path in paths:
    os.makedirs(os.path.join(args.load_net, path, 'predictions'), exist_ok=True)
    for p in sorted(os.listdir(path)):
        if 'pred' not in p:
            if 'wide' in  path.lower():
                args.net = 'WideResNet'
                # Here the net path is saved like '/path/to/saved/models/WideResNet_10.pth'
                args.widen_factor = int(path.split('/')[-1].split('.')[0].split('_')[1])
            else:
                args.net = path.split('/')[-1].split('.')[0]
            net = get_model(args, device)
            temp_path = os.path.join(path,p)
            net.load_state_dict(torch.load(temp_path))
            pred_arr = []
            for run in range(args.epochs):
                random.seed(a=(args.set_data_seed+run), version=2)
                images, labels, image_ids = get_random_images(testloader.dataset)
                planeloader = make_planeloader(images, args)
                preds = decision_boundary(args, net, planeloader, device)
                pred_arr.append(torch.stack(preds).argmax(1).cpu())
            torch.save(pred_arr, os.path.join(args.load_net, path, 'predictions') + '/' + temp_path.split('/')[-1].split('.pth')[0] + '_preds.pth')
            

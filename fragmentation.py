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
import itertools

import wandb
from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
# from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
# from utils import simple_lapsed_time

args = options().parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)
net.load_state_dict(torch.load(args.load_net))
net.eval()


l_all = np.arange(len(trainloader.dataset.targets))
l_test = np.arange(len(testloader.dataset.targets))

if args.active_log:
    wandb.init(project="dd_fragmentation_db", name = f'frag_{args.net}_{args.set_seed}')
    wandb.config.update(args)
from data import make_planeloader
from evaluation import decision_boundary
from db_quant_utils import num_connected_components,rel_index

# import ipdb; ipdb.set_trace()
alltrain_alltrain = num_connected_components(l_all,l_all,trainloader,trainloader,args.epochs,net,device,args)
alltest_alltest = num_connected_components(l_test,l_test,testloader,testloader,args.epochs,net,device,args)

perclass_samples = args.epochs//10

ctrain_ctrain = []
ctrain_asgtest = []

for class_index in range(10):
    if args.active_log:
                wandb.log({'inclass':class_index})
    l_all_train, l_mis_cls, l_corr_cls,l_all_test,l_fromcls_mislab = rel_index(class_index,trainloader.dataset,testloader.dataset,0)
    ctrain_ctrain.append(num_connected_components(l_corr_cls,l_corr_cls,trainloader,trainloader,perclass_samples,net,device,args))
    # ctrain_asgtest.append(num_connected_components(l_corr_cls,l_all_test,trainloader,testloader,perclass_samples,net,device,args))

ctrain_ctrain =list(itertools.chain(*ctrain_ctrain))
# ctrain_asgtest =list(itertools.chain(*ctrain_asgtest))


mean_fragmentation = {
        'all' : np.mean(alltrain_alltrain),
        'correct' : np.mean(ctrain_ctrain),
        'test' : np.mean(alltest_alltest),
        'all_std' : np.std(alltrain_alltrain),
        'correct_std' : np.std(ctrain_ctrain),
        'test_std' : np.std(alltest_alltest)
        # 'ctrain_asgtest' : np.mean(ctrain_asgtest)
}

if args.active_log:
    wandb.log(mean_fragmentation)
else:
    print(mean_fragmentation)
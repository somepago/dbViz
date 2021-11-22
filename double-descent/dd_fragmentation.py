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
import itertools

from models import *
from db_quant_utils import rel_index



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
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args.eval = True

if args.mixup:
    args.model_path = './checkpoint/mixup'
else:
    args.model_path = f'./checkpoint/{args.set_seed}/{args.set_data_seed}'

if args.active_log:
    import wandb
    if not args.mixup: 
        wandb.init(project="dd_fragmentation", name = f'frag_{args.noise_rate}noise_{args.k}k_{args.set_seed}')
    else:
        wandb.init(project="dd_fragmentation", name = f'frag_{args.noise_rate}noise_{args.k}k__{args.set_seed}_wmixup')
    wandb.config.update(args)


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

l_mis = np.where(np.array(trainset_noisy.targets) != np.array(trainset_noisy.true_targets))[0] #mislabeled images in noisy case
l_corr = np.where(np.array(trainset_noisy.targets) == np.array(trainset_noisy.true_targets))[0] #correctly labeled images in noisy case
l_all = np.arange(len(trainset.targets))
l_test = np.arange(len(testset.targets))


from db.data import make_planeloader
from db.evaluation import decision_boundary
from db.utils import connected_components


def num_connected_components(dlist1,dlist2, loader1,loader2, num_samples,net,device,args):
    cc_list = []
    for i in range(num_samples):
        # import ipdb; ipdb.set_trace()
        dirlist = np.random.choice(dlist1, 2)
        dirlist2 = np.random.choice(dlist2, 1)
        images = [loader1.dataset[j][0] for j in dirlist]
        images.append(loader2.dataset[dirlist2[0]][0])
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        ncc = connected_components(preds,args)
        cc_list.append(ncc)
        # print(dirlist,dirlist2,ncc)
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




alltrain_alltrain = num_connected_components(l_all,l_all,trainloader,trainloader,args.num_samples,net,device,args)
alltest_alltest = num_connected_components(l_test,l_test,testloader,testloader,args.num_samples,net,device,args)

perclass_samples = args.num_samples//10

ctrain_ctrain = []
# ctrain_asgtest = []

for class_index in range(10):
    if args.active_log:
                wandb.log({'inclass':class_index})
    l_all_train, l_mis_cls, l_corr_cls,l_all_test,l_fromcls_mislab = rel_index(class_index,trainset_noisy,testset)
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
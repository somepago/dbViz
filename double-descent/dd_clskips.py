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
import itertools
import pickle
from models import *



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--k', default=64, type=int, help='depth of model')      
parser.add_argument('--noise_rate', default=0.2, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--model_path', type=str, default='./checkpoint')
parser.add_argument('--mixup', action='store_true') 


parser.add_argument('--plot_method', default='train', type=str)
# parser.add_argument('--plot_train_class', default=4, type=int)

# parser.add_argument('--plot_train_imgs', action='store_true')
# parser.add_argument('--plot_path', type=str, default='./imgs')
# parser.add_argument('--extra_path', type=str, default=None)
# parser.add_argument('--imgs', default=None,
#                             type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')

parser.add_argument('--active_log', action='store_true') 

parser.add_argument('--num_samples', default=1 , type=int)
parser.add_argument('--num_iters', default=10 , type=int)
parser.add_argument('--delta', default=0.5, type=float, help='how far from image')

parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
# parser.add_argument('--run_name', type=str, default='temp')    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


if args.mixup:
    args.model_path = './checkpoint/mixup'
    
if args.active_log:
    import wandb
    if not args.mixup: 
        wandb.init(project="dd_quantif_clskips", name = f'ddskips_{args.plot_method}_{args.noise_rate}noise_{args.k}k')
    else:
        wandb.init(project="dd_quantif_clskips", name = f'ddskips_{args.plot_method}_{args.noise_rate}noise_{args.k}k_wmixup')

    wandb.config.update(args)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
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
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


print('RAN FINE TILL HERE!')
def rel_index(args,class_index):

    l_all_train = np.where(np.array(trainset.targets) == class_index)[0]
    l_mis_cls = 0
    l_corr_cls = l_all_train
    if args.noise_rate > 0:
        l_mis = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0] #mislabeled images
        l_corr = np.where(np.array(trainset.targets) == np.array(trainset.true_targets))[0] #correctly labeled images
        l_mis_cls = np.intersect1d(l_all_train,l_mis)
        l_corr_cls = np.intersect1d(l_all_train,l_corr)
    l_all_test = np.where(np.array(testset.targets) == class_index)[0]

    return l_all_train, l_mis_cls, l_corr_cls,l_all_test


def count_skips(dlist,loader,num_steps, num_samples,net,device):
    skips_list = []
    for ns in range(num_samples):
        # print(f'Sample number: {ns}')
        baselist = np.random.choice(dlist, 2)
        # baselist = np.array([826, 2272])
        img1 = loader.dataset[baselist[0]][0].unsqueeze(0).expand(num_steps,-1,-1,-1)
        img2 = loader.dataset[baselist[1]][0].unsqueeze(0).expand(num_steps,-1,-1,-1)
        img1 = img1.to(device)
        img2 = img2.to(device)
        alphas = torch.from_numpy(np.linspace(0,1,num_steps)).reshape(num_steps,1,1,1)
        alphas  = alphas.to(device)
        img_batch = img1 + alphas*(img2-img1)
        img_batch = img_batch.to(device=device, dtype=torch.float)
        preds = torch.argmax(net(img_batch),dim=1).cpu().numpy()
        # print(preds)
        skip = (np.diff(preds)!=0).sum()
        skips_list.append(skip)

    # print(skips_list)
    return skips_list


 

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



if args.plot_method == 'train':
    loader_f = trainloader
else:
    loader_f =  testloader
skipcounts_dict = {}
meanskips = {}
allskips = []
for i in range(10):
    alist,_,_,_ = rel_index(args,i)
    sl = count_skips(alist,loader_f,args.num_iters, args.num_samples,net,device) 
    skipcounts_dict[i] = sl
    allskips.append(sl)
    meanskips[f'class_{i}'] = np.mean(sl) 
    if args.active_log:
        wandb.log({'inclass':i})
merged = list(itertools.chain(*allskips))
meanskips[f'all'] = np.mean(merged) 
if args.active_log:
    wandb.log(meanskips)

if args.mixup:
    savepath = f'ddquant/clskips_wmixup/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'
else:
    savepath = f'ddquant/clskips/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'

tdir = '/'.join([p for p in (savepath.split('/'))[:-1]])
os.makedirs(tdir, exist_ok=True)

pickle.dump( skipcounts_dict, open( f"{savepath}.p", "wb" ) )
    

# else:
#     if args.plot_method == 'train':
#         unique_classes = {
#             'aa' : avg_margin(l_all,l_all,trainloader,args.num_iters, args.num_samples,net,device)
#         } 
#         if args.active_log:
#                     wandb.log({
#                         'aa': np.mean(unique_classes['aa']),
#                         'len_aa' : len(unique_classes['aa'])
#                 })
#     else:
#         unique_classes = {
#             'aa' : avg_margin(l_all,l_all,testloader,args.num_iters, args.num_samples,net,device)
#         } 
#         if args.active_log:
#                     wandb.log({
#                         'aa': np.mean(unique_classes['aa']),
#                         'len_aa' : len(unique_classes['aa'])
#                 })



# import json
# with open(f"{savepath}.txt", 'w') as f: 
#     f.write(json.dumps(unique_classes))


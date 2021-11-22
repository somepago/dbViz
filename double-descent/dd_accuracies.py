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
parser.add_argument('--model_path', type=str, default='./checkpoint')
parser.add_argument('--mixup', action='store_true') 


parser.add_argument('--plot_method', default='mislabwithtrue', type=str)
# parser.add_argument('--plot_train_class', default=4, type=int)

# parser.add_argument('--plot_train_imgs', action='store_true')
# parser.add_argument('--plot_path', type=str, default='./imgs')
# parser.add_argument('--extra_path', type=str, default=None)
# parser.add_argument('--imgs', default=None,
#                             type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')

parser.add_argument('--active_log', action='store_true') 

# parser.add_argument('--num_samples', default=1 , type=int)
# parser.add_argument('--num_iters', default=10 , type=int)

parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
# parser.add_argument('--run_name', type=str, default='temp')    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args.eval = True
if args.mixup:
    args.model_path = './checkpoint/mixup'
if args.active_log:
    import wandb
    if not args.mixup: 
        wandb.init(project="dd_quantif_accuracies", name = f'ddmargin_{args.noise_rate}noise_{args.k}k')
    else:
        wandb.init(project="dd_quantif_accuracies", name = f'ddmargin_{args.noise_rate}noise_{args.k}k_wmixup')

    wandb.config.update(args)


print('==> Preparing data..')
transform_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

torch.manual_seed(args.set_data_seed)
if args.noise_rate > 0: 
    from data_noisy import cifar10Nosiy
    trainset = cifar10Nosiy(root='./data', train=True,transform=transform_data, download=True,
                                            asym=args.asym,
                                            nosiy_rate=args.noise_rate)
    
else:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_data)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_data)

# if args.plot_method =='train':
#     loader_data_all = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
# else:
#     loader_data_all = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')


print('RAN FINE TILL HERE!')
if args.noise_rate > 0:
    l_mis = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0] #mislabeled images
    l_corr = np.where(np.array(trainset.targets) == np.array(trainset.true_targets))[0] #correctly labeled images
else:
    l_mis = None
    l_corr = None
l_all = np.arange(len(trainset.targets))
l_all_test = np.arange(len(testset.targets))


mp = args.model_path
modelpath = f'{mp}/dd_Adam_{args.noise_rate}noise_{args.k}k/ckpt.pth'
net = make_resnet18k(args.k)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(modelpath)
checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['net'])
net.eval()


def acc_calc(loader,net,device):
    # import ipdb; ipdb.set_trace()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    return test_acc

def acc_wo_loader(datalist,dset,net,device):
    loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in datalist:
            img = loader.dataset[i][0].unsqueeze(0)
            img = img.to(device)
            true_label = dset.true_targets[i]
            outputs = net(img)
            _, predicted = outputs.max(1)
            total+=1
            correct += predicted.eq(true_label).sum().item()
    test_acc = correct/total
    return test_acc


acc_list = []
for datalist,dset in [(l_mis,trainset),(l_corr,trainset),(l_all,trainset),(l_all_test,testset)]:
    if datalist is not None:
        temp_set  = torch.utils.data.Subset(dset, datalist)
        temp_loader = torch.utils.data.DataLoader(temp_set, batch_size=128, shuffle=False, num_workers=4)
        acc = acc_calc(temp_loader,net,device)
    else:
        acc = 0
    acc_list.append(acc)

if l_mis is not None:
    tac = acc_wo_loader(l_mis,trainset,net,device)
else:
    tac = 0

acc_dict = {
    'mislabeled_acc': acc_list[0],
    'mislabeled_acc_truelab' : tac,
    'correctlabels_acc' : acc_list[1],
    'all_acc' : acc_list[2],
    'test_acc' : acc_list[3]
}

print(acc_dict)
if args.active_log:
    wandb.log(acc_dict)

# if args.noise_rate > 0 and args.plot_method == 'train':
#     unique_classes = {
#         # 'cc' : avg_margin(l_corr,l_corr,trainloader,args.num_iters, args.num_samples,net,device),
#         # 'mm' : avg_margin(l_mis,l_mis,trainloader,args.num_iters, args.num_samples,net,device),
#         # 'mc' : avg_margin(l_mis,l_corr,trainloader,args.num_iters, args.num_samples,net,device),
#         'ca' : avg_margin(l_corr,l_all,trainloader,args.num_iters, args.num_samples,net,device),
#         'ma' : avg_margin(l_mis,l_all,trainloader,args.num_iters, args.num_samples,net,device),
#         'aa' : avg_margin(l_all,l_all,trainloader,args.num_iters, args.num_samples,net,device)
#     } 
#     if args.active_log:
#                 wandb.log({
#                     'aa': np.mean(unique_classes['aa']),
#                     'ca' : np.mean(unique_classes['ca']),
#                     'ma' : np.mean(unique_classes['ma']),
#                     'aa_median': np.median(unique_classes['aa']),
#                     'ca_median' : np.median(unique_classes['ca']),
#                     'ma_median' : np.median(unique_classes['ma']),
#                     'len_aa' : len(unique_classes['aa']),
#                     'len_ca' : len(unique_classes['ca']),
#                     'len_ma' : len(unique_classes['ma']

#                 )})
#     else:
#         print({
#                     'aa': np.mean(unique_classes['aa']),
#                     'ca' : np.mean(unique_classes['ca']),
#                     'ma' : np.mean(unique_classes['ma']),
#                     'aa_median': np.median(unique_classes['aa']),
#                     'ca_median' : np.median(unique_classes['ca']),
#                     'ma_median' : np.median(unique_classes['ma']),
#                     'len_aa' : len(unique_classes['aa']),
#                     'len_ca' : len(unique_classes['ca']),
#                     'len_ma' : len(unique_classes['ma']

#                 )})

# else:
#     if args.plot_method == 'train':
#         unique_classes = {
#             'aa' : avg_margin(l_all,l_all,trainloader,args.num_iters, args.num_samples,net,device)
#         } 
#         if args.active_log:
#                     wandb.log({
#                         'aa': np.mean(unique_classes['aa']),
#                         'aa_median': np.median(unique_classes['aa']),
#                         'len_aa' : len(unique_classes['aa'])
#                 })
#     else:
#         unique_classes = {
#             'aa' : avg_margin(l_all,l_all,testloader,args.num_iters, args.num_samples,net,device)
#         } 
#         if args.active_log:
#                     wandb.log({
#                         'aa': np.mean(unique_classes['aa']),
#                         'aa_median': np.median(unique_classes['aa']),
#                         'len_aa' : len(unique_classes['aa'])
#                 })


# if args.mixup:
#     savepath = f'ddquant/margin_wmixup/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'
# else:
#     savepath = f'ddquant/margin/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'

# tdir = '/'.join([p for p in (savepath.split('/'))[:-1]])
# os.makedirs(tdir, exist_ok=True)

# import json
# with open(f"{savepath}.txt", 'w') as f: 
#     f.write(json.dumps(unique_classes))
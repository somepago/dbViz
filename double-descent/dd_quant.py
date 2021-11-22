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


# parser.add_argument('--plot_method', default='train', type=str)
# parser.add_argument('--plot_train_class', default=4, type=int)

# parser.add_argument('--plot_train_imgs', action='store_true')
# parser.add_argument('--plot_path', type=str, default='./imgs')
# parser.add_argument('--extra_path', type=str, default=None)
# parser.add_argument('--imgs', default=None,
#                             type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')

parser.add_argument('--active_log', action='store_true') 

parser.add_argument('--num_samples', default=100 , type=int)
parser.add_argument('--num_iters', default=10 , type=int)
parser.add_argument('--delta', default=0.5, type=float, help='how far from image')

parser.add_argument('--set_seed', default=1 , type=int)
parser.add_argument('--set_data_seed', default=1 , type=int)
# parser.add_argument('--run_name', type=str, default='temp')    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

if args.active_log:
    import wandb
    wandb.init(project="dd_quantif", name = f'dd_{args.delta}delta_{args.noise_rate}noise_{args.k}k')
    wandb.config.update(args)


best_acc = 0  # best test accuracy
best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if not os.path.isdir('./checkpoint'):
    os.mkdir('./checkpoint')
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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

def image_atdelta(img1,img2,delta):
    a = img2 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    b = img1 + delta*a
    return b

def num_unique_classes_atdelta(dlist1,dlist2,loader,num_iters, num_samples,delta,net,device):
    baselist = np.random.choice(dlist1, num_samples)
    unique_classes = []
    for i in baselist:
        dirlist = np.random.choice(dlist2, num_iters)
        preds = []
        # import ipdb; ipdb.set_trace()
        for j in dirlist:
            img1 = loader.dataset[i][0].unsqueeze(0)
            img2 = loader.dataset[j][0].unsqueeze(0)
            imdelta = image_atdelta(img1,img2,delta)
            imdelta = imdelta.to(device)
            output = torch.argmax(net(imdelta))
            preds.append(output.item())
        img1 = img1.to(device)
        # import ipdb; ipdb.set_trace()
        preds.append(torch.argmax(net(img1)).item())
        unique_classes.append(len(np.unique(preds)))
    return unique_classes
        

modelpath = f'./checkpoint/dd_Adam_{args.noise_rate}noise_{args.k}k/ckpt.pth'
net = make_resnet18k(args.k)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(modelpath)
net.load_state_dict(checkpoint['net'])

# import time
# start = time.time()

if args.noise_rate > 0:
    unique_classes = {
        'cc' : num_unique_classes_atdelta(l_corr,l_corr,trainloader,args.num_iters, args.num_samples,args.delta,net,device),
        'mm' : num_unique_classes_atdelta(l_mis,l_mis,trainloader,args.num_iters, args.num_samples,args.delta,net,device),
        'mc' : num_unique_classes_atdelta(l_mis,l_corr,trainloader,args.num_iters, args.num_samples,args.delta,net,device),
        'ca' : num_unique_classes_atdelta(l_corr,l_all,trainloader,args.num_iters, args.num_samples,args.delta,net,device),
        'ma' : num_unique_classes_atdelta(l_mis,l_all,trainloader,args.num_iters, args.num_samples,args.delta,net,device),
        'aa' : num_unique_classes_atdelta(l_all,l_all,trainloader,args.num_iters, args.num_samples,args.delta,net,device)
    } 
    if args.active_log:
                wandb.log({
                    'aa': np.mean(unique_classes['aa']),
                    'cc' : np.mean(unique_classes['cc']),
                    'mm' : np.mean(unique_classes['mm']),
                    'mc' : np.mean(unique_classes['mc']),
                    'ca' : np.mean(unique_classes['ca']),
                    'ma' : np.mean(unique_classes['ma']

                )})

else:
    unique_classes = {
        'aa' : num_unique_classes_atdelta(l_all,l_all,trainloader,args.num_iters, args.num_samples,args.delta,net,device)
    } 
    if args.active_log:
                wandb.log({'aa': np.mean(unique_classes['aa'])
            })

# end = time.time()
# time_taken = end - start
# print('Time: ',time_taken) 
# if args.active_log:
#                 wandb.log({ 'Time' : time_taken
#             })

savepath = f'ddquant/{args.noise_rate}nr{args.k}k_{args.delta}del_{args.num_samples}sampl_{args.num_iters}iters'
import json
with open(f"{savepath}.txt", 'w') as f: 
    f.write(json.dumps(unique_classes))

# if args.plot_method == 'test':
#     image_ids = args.imgs
#     images = [testloader.dataset[i][0] for i in image_ids]
#     labels = [testloader.dataset[i][1] for i in image_ids]
# else:
#     if args.plot_method == 'train_ids':
#         image_ids = args.imgs
#     else:
#         train_class = args.plot_train_class
#         img_locs = args.imgs
#         if args.noise_rate > 0:
#             # import ipdb; ipdb.set_trace()
#             l1 = np.where(np.array(trainset.true_targets) == train_class)[0]
#             l2 = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0]
#             inc_id = np.intersect1d(l1,l2)[img_locs[0]]
#             corr_id = np.setdiff1d(l1,l2)[img_locs[1:]]
#             image_ids = [corr_id[0],corr_id[1],inc_id]
#             if args.plot_method == 'train_corr':
#                 image_ids[-1] = np.setdiff1d(l1,l2)[img_locs[0]]
#             elif args.plot_method == 'train_2inc':
#                 image_ids[1] = np.intersect1d(l1,l2)[img_locs[-1]]
#         else:
#             l1 = np.where(np.array(trainset.true_targets) == train_class)
#             image_ids = l1[img_locs]
#     images = [trainloader.dataset[i][0] for i in image_ids]
#     labels = [trainloader.dataset[i][1] for i in image_ids]

# print(image_ids, labels)
# sampleids = '_'.join(list(map(str,image_ids)))




# from db.data import make_planeloader
# from db.evaluation import decision_boundary
# from db.utils import produce_plot_alt,produce_plot_x
# # import ipdb; ipdb.set_trace()
# planeloader = make_planeloader(images, args)
# preds = decision_boundary(args, net, planeloader, device)
# plot_path = f'{args.plot_path}/{sampleids}/rn_{args.k:02d}' 
# plot = produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader, temp=args.temp)
# plot = produce_plot_x(plot_path, preds, planeloader, images, labels, trainloader, temp=args.temp)
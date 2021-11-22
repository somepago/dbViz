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

args.eval = True
if args.mixup:
    args.model_path = './checkpoint/mixup'
if args.active_log:
    import wandb
    if not args.mixup: 
        wandb.init(project="dd_quantif_margin", name = f'ddmargin_{args.plot_method}_{args.noise_rate}noise_{args.k}k')
    else:
        
        wandb.init(project="dd_quantif_margin", name = f'ddmargin_{args.plot_method}_{args.noise_rate}noise_{args.k}k_wmixup')

    wandb.config.update(args)

# best_acc = 0  # best test accuracy
# best_epoch = 0
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# if not os.path.isdir('./checkpoint'):
#     os.mkdir('./checkpoint')
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
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


print('RAN FINE TILL HERE!')
if args.plot_method == 'train':
    if args.noise_rate > 0:
        l_mis = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0] #mislabeled images
        l_corr = np.where(np.array(trainset.targets) == np.array(trainset.true_targets))[0] #correctly labeled images
    l_all = np.arange(len(trainset.targets))
else:
    l_all = np.arange(len(testset.targets))

def image_atdelta(img1,img2,delta):
    a = img2 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    b = img1 + delta*a
    return b

def old_avg_margin(dlist1,dlist2,loader,num_iters, num_samples,net,device):
    print(f'In the case of list lengths: {len(dlist1)},{len(dlist2)}')
    baselist = np.random.choice(dlist1, num_samples)
    mean_margins = []
    for i in baselist:
        # print(f'The image: {i}')
        dirlist = np.random.choice(dlist2, num_iters+10)
        margins = []
        img = loader.dataset[i][0].unsqueeze(0)
        img = img.to(device)
        img_op = torch.argmax(net(img)).item()
        iter_count = 0
        counts = []
        while len(margins) < num_iters and iter_count < num_iters+10:
            j = dirlist[iter_count]
            iter_count+=1
            img1 = loader.dataset[i][0].unsqueeze(0)
            img2 = loader.dataset[j][0].unsqueeze(0)
            delta = 0.01
            imdelta = image_atdelta(img1,img2,delta)
            imdelta = imdelta.to(device)
            output = torch.argmax(net(imdelta)).item()
            count = 0
            step = 0.01
            while (img_op ==output) and count <=100:
                delta+= step
                imdelta = image_atdelta(img1,img2,delta)
                imdelta = imdelta.to(device)
                output = torch.argmax(net(imdelta)).item()
                if 10 < count < 20:
                    step = 0.1
                elif 20 <= count or count <= 30:
                    step = 0.5
                elif 30 < count or count <= 50:
                    step = 1
                elif 50 < count or count <= 75:
                    step = 10
                elif 75 < count or count <= 100:
                    step = 100
                count+=1
            # import ipdb; ipdb.set_trace()
            counts.append(count)
            if count <= 100:
                margins.append(delta)
                
        # import ipdb; ipdb.set_trace()
        # print(counts)
        if len(margins) >= num_iters//3:
            mean_margins.append(np.mean(margins))
        pdone = len(mean_margins)
        if pdone%1000 ==0:
            print(f'At {pdone}th point')
    return mean_margins
        

def delta_counter(max_steps):
    delta_list = [0]
    delta = 0
    step = 0.01
    for count in range(max_steps-1):
        # print(count,step)
        if count > 10 and count < 20:
            step = 0.1
        elif count >= 20 and count <= 50:
            step = 0.5
        elif count > 50  and count <= 75:
            step = 1
        elif count > 75 and count <= 100:
            step = 10
        elif count > 100:
            step = 20
        delta+=step
        delta_list.append(delta)
#     print(delta_list)
    return np.array(delta_list)


def avg_margin(dlist1,dlist2,loader,num_iters, num_samples,net,device,max_steps=200):
    print(f'In the case of list lengths: {len(dlist1)},{len(dlist2)}')
    baselist = np.random.choice(dlist1, num_samples)
    deltas = torch.from_numpy(delta_counter(max_steps).reshape(max_steps,1,1,1))
    deltas = deltas.to(device)
    deltas_temp = torch.squeeze(deltas)
    mean_margins = []
    for i in baselist:
        # print(f'The image: {i}')
        dirlist = np.random.choice(dlist2, num_iters+10)
        margins = []
        iter_count = 0
        while len(margins) < num_iters and iter_count < num_iters+10:
            j = dirlist[iter_count]
            iter_count+=1
            img1 = loader.dataset[i][0].unsqueeze(0)
            img2 = loader.dataset[j][0].unsqueeze(0)
            a = img2 - img1
            a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
            a = a / a_norm
            img1 = loader.dataset[i][0].unsqueeze(0).expand(max_steps,-1,-1,-1)
            a = a.expand(max_steps,-1,-1,-1)
            img1 = img1.to(device)
            a = a.to(device)
            # import ipdb; ipdb.set_trace()
            img_batch = img1 + deltas*a
            img_batch = img_batch.to(device=device, dtype=torch.float)
            preds = torch.argmax(net(img_batch),dim=1).cpu().numpy()
            where_db = np.where(np.diff(preds) != 0)[0]
            if where_db.size !=0:
                delta = deltas_temp[where_db[0]].item()
                margins.append(delta)
        if len(margins) >= num_iters//2:
            mean_margins.append(np.mean(margins))
        pdone = len(mean_margins)
        if pdone%1000 ==0:
            print(f'At {pdone}th point')
    return mean_margins

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

# import time
# start = time.time()
# cc = avg_margin(l_corr,l_all,trainloader,args.num_iters, args.num_samples,net,device)
# print(cc)
# end  = time.time()
# print(f'Time taken: {end-start}')

if args.noise_rate > 0 and args.plot_method == 'train':
    unique_classes = {
        # 'cc' : avg_margin(l_corr,l_corr,trainloader,args.num_iters, args.num_samples,net,device),
        # 'mm' : avg_margin(l_mis,l_mis,trainloader,args.num_iters, args.num_samples,net,device),
        # 'mc' : avg_margin(l_mis,l_corr,trainloader,args.num_iters, args.num_samples,net,device),
        'ca' : avg_margin(l_corr,l_all,trainloader,args.num_iters, args.num_samples,net,device),
        'ma' : avg_margin(l_mis,l_all,trainloader,args.num_iters, args.num_samples,net,device),
        'aa' : avg_margin(l_all,l_all,trainloader,args.num_iters, args.num_samples,net,device)
    } 
    if args.active_log:
                wandb.log({
                    'aa': np.mean(unique_classes['aa']),
                    'ca' : np.mean(unique_classes['ca']),
                    'ma' : np.mean(unique_classes['ma']),
                    'aa_median': np.median(unique_classes['aa']),
                    'ca_median' : np.median(unique_classes['ca']),
                    'ma_median' : np.median(unique_classes['ma']),
                    'len_aa' : len(unique_classes['aa']),
                    'len_ca' : len(unique_classes['ca']),
                    'len_ma' : len(unique_classes['ma']

                )})
    else:
        print({
                    'aa': np.mean(unique_classes['aa']),
                    'ca' : np.mean(unique_classes['ca']),
                    'ma' : np.mean(unique_classes['ma']),
                    'aa_median': np.median(unique_classes['aa']),
                    'ca_median' : np.median(unique_classes['ca']),
                    'ma_median' : np.median(unique_classes['ma']),
                    'len_aa' : len(unique_classes['aa']),
                    'len_ca' : len(unique_classes['ca']),
                    'len_ma' : len(unique_classes['ma']

                )})

else:
    if args.plot_method == 'train':
        unique_classes = {
            'aa' : avg_margin(l_all,l_all,trainloader,args.num_iters, args.num_samples,net,device)
        } 
        if args.active_log:
                    wandb.log({
                        'aa': np.mean(unique_classes['aa']),
                        'aa_median': np.median(unique_classes['aa']),
                        'len_aa' : len(unique_classes['aa'])
                })
    else:
        unique_classes = {
            'aa' : avg_margin(l_all,l_all,testloader,args.num_iters, args.num_samples,net,device)
        } 
        if args.active_log:
                    wandb.log({
                        'aa': np.mean(unique_classes['aa']),
                        'aa_median': np.median(unique_classes['aa']),
                        'len_aa' : len(unique_classes['aa'])
                })


if args.mixup:
    savepath = f'ddquant/margin_wmixup/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'
else:
    savepath = f'ddquant/margin/{args.plot_method}/{args.noise_rate}nr{args.k}k_{args.num_samples}sampl_{args.num_iters}iters'

tdir = '/'.join([p for p in (savepath.split('/'))[:-1]])
os.makedirs(tdir, exist_ok=True)

import json
with open(f"{savepath}.txt", 'w') as f: 
    f.write(json.dumps(unique_classes))
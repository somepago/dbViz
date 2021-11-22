'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import argparse
import numpy as np
from models import *
from db.data import make_planeloader
from db.evaluation import decision_boundary
from db.utils import produce_plot_alt, mixup_data
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=200, type=int, help='total number of training epochs')      
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--k', default=64, type=int, help='depth of model')      
parser.add_argument('--active_log', action='store_true') 
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--noise_rate', default=0.2, type=float, help='label noise')
parser.add_argument('--asym', action='store_true')
parser.add_argument('--mixup', action='store_true') 
parser.add_argument('--mixup_alpha', default=1.0, type=float, help='hyperparameter alpha for mixup')


parser.add_argument('--dd_epoch_study', action='store_true') 
parser.add_argument('--plot_animation', action='store_true') 
parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
parser.add_argument('--range_l', default=0.5, type=float, help='how far `left` to go in the plot')
parser.add_argument('--range_r', default=0.5, type=float, help='how far `right` to go in the plot')
parser.add_argument('--temp', default=5.0, type=float)
parser.add_argument('--plot_method', default='test', type=str)
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
if args.active_log:
    import wandb
    if not args.mixup:
        wandb.init(project="double_descent", name = f'dd_{args.opt}_{args.noise_rate}noise_{args.k}k')
    else:
        wandb.init(project="double_descent", name = f'dd_{args.opt}_{args.noise_rate}noise_{args.k}k_wmixup')
    wandb.config.update(args)

best_acc = 0  # best test accuracy
best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if not os.path.isdir('./checkpoint'):
    os.mkdir('./checkpoint')
if not args.mixup:
    save_path = f'checkpoint/{args.set_seed}/{args.set_data_seed}/dd_{args.opt}_{args.noise_rate}noise_{args.k}k'
else:
    save_path = f'checkpoint/mixup/dd_{args.opt}_{args.noise_rate}noise_{args.k}k'

if not os.path.isdir(save_path):
    os.makedirs(save_path, exist_ok=True)

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

if args.imgs is not None:
    if args.plot_method == 'test':
        image_ids = args.imgs
        images = [testloader.dataset[i][0] for i in image_ids]
        labels = [testloader.dataset[i][1] for i in image_ids]
    else:
        if args.noise_rate > 0:
            image_ids = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0][:3]
        else:
            image_ids = [10, 500,8000]
        images = [trainloader.dataset[i][0] for i in image_ids]
        labels = [trainloader.dataset[i][1] for i in image_ids]

    print(image_ids, labels)
    sampleids = '_'.join(list(map(str,image_ids)))
    planeloader = make_planeloader(images, args)

# Model
print('==> Building model..')
torch.manual_seed(args.set_seed)
net = make_resnet18k(args.k)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.set_seed}/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if not args.mixup:
    criterion = nn.CrossEntropyLoss()
else:
    from db.utils import mixup_criterion
    criterion = mixup_criterion
test_criterion = nn.CrossEntropyLoss()

if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def train_mixup(epoch,args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha, True)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(None, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch,args):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = test_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_acc = correct/total
        if args.active_log:
            wandb.log({'epoch': epoch ,'test_error': 1-test_acc
        })
            
    if args.plot_animation:
        if epoch%2 == 0:
            preds = decision_boundary(args, net, planeloader, device)
            plot_path  =  f'{args.plot_path}/{args.plot_method}/{sampleids}/{args.noise_rate}/{args.k:02d}/{epoch:03d}' 
            produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader, temp=args.temp)

    # Save checkpoint.
    acc = 100.*correct/total
    if epoch%100==0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        torch.save(state, f'{save_path}/ckpt.pth')
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

    if args.dd_epoch_study:
        if args.k > 20:
            epoch_list = [1,5,10,16,21,24,29,42,52,58,73,85,96,127,212,430,455,504]
        else:
             epoch_list = [1,5,10,16,21,24,29,42,52,58,73,85,96,127,212,430,455,458,504,571,883,1000]
        if epoch in epoch_list:
            state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            }
            torch.save(state, f'{save_path}/ckpt_{epoch}.pth')

for epoch in range(start_epoch, start_epoch+args.epochs):
    if not args.mixup:
        train(epoch)
    else:
        train_mixup(epoch,args)

    test(epoch,args)
    if args.opt == "SGD":
        scheduler.step()



if args.active_log:
            wandb.log({'best_epoch': best_epoch ,'best_acc': best_acc
        })
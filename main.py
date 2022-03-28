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

from model import get_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv
from options import options
from utils import simple_lapsed_time

args = options().parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = args.save_net
if args.active_log:
    import wandb
    idt = '_'.join(list(map(str,args.imgs)))
    wandb.init(project="decision_boundaries", name = '_'.join([args.net,args.train_mode,idt,'seed'+str(args.set_seed)]) )
    wandb.config.update(args)

# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)

test_acc, predicted = test(args, net, testloader, device, 0)
print("scratch prediction ", test_acc)

criterion = get_loss_function(args)
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(args, optimizer)

elif args.opt == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)



# Train or load base network
print("Training the network or loading the network")

start = time.time()
best_acc = 0  # best test accuracy
best_epoch = 0
if args.load_net is None:
    if args.plot_animation:
        image_ids = args.imgs
        sampleids = '_'.join(list(map(str,image_ids)))
        os.makedirs(f'images/{args.net}/{args.train_mode}/{sampleids}/{str(args.set_seed)}', exist_ok=True)
        args.plot_path = os.path.join('images', args.net, args.train_mode, sampleids,str(args.set_seed))
        if args.extra_path != None:
            os.makedirs(f'images/{args.net}/{args.train_mode}/{sampleids}/{args.extra_path}/{str(args.set_seed)}', exist_ok=True)
            args.plot_path = os.path.join('images', args.net, args.train_mode, sampleids, args.extra_path, str(args.set_seed))

        if args.imgs is None:
            #images, labels = get_random_images(trainloader.dataset)
            images, labels = get_random_images(testloader.dataset)
        elif -1 in args.imgs:
            #LF maybe move farther up? 
            torch.manual_seed(args.set_data_seed)
            dummy_imgs, _, _ = get_random_images(testloader.dataset)
            images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
        elif -10 in args.imgs:
            image_ids = args.imgs[0]
            images = [testloader.dataset[image_ids][0]]
            labels = [testloader.dataset[image_ids][1]]
            for i in list(range(2)):
                temp = torch.zeros_like(images[0])
                if i == 0:
                    temp[0,0,0] = 1
                else:
                    temp[0,-1,-1] = 1

                images.append(temp)
                labels.append(0)

        else:
            image_ids = args.imgs
            images = [testloader.dataset[i][0] for i in image_ids]
            labels = [testloader.dataset[i][1] for i in image_ids]
            print(labels)
        if args.adv:
            adv_net = AttackPGD(net, trainloader.dataset)
            adv_preds, imgs = adv_net(torch.stack(images).to(device), torch.tensor(labels).to(device), targeted=args.targeted)
            images = [img.cpu() for img in imgs]

        print(labels)
        planeloader = make_planeloader(images, args)
        print(len(planeloader))
    for epoch in range(args.epochs):
        train_acc = train(args, net, trainloader, optimizer, criterion, device, args.train_mode, sam_radius=args.sam_radius)
        if args.plot_animation:
            test_acc, predicted = test(args, net, testloader, device, epoch,images,labels,planeloader)
        else:
            test_acc, predicted = test(args, net, testloader, device, epoch)
        print(f'EPOCH:{epoch}, Test acc: {test_acc}')
        if args.active_log:
            wandb.log({'epoch': epoch ,'test_accuracy': test_acc
            })
        if args.dryrun:
            break
        if args.opt == 'SGD':
            scheduler.step()

        # Save checkpoint.
        if test_acc > best_acc:
            print(f'The best epoch is: {epoch}')
            os.makedirs(f'saved_models/{args.train_mode}/{str(args.set_seed)}', exist_ok=True)
            if args.extra_path != None:
                os.makedirs(save_path, exist_ok=True)
                print(f'{save_path}/{args.save_net}.pth')
                if torch.cuda.device_count() > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()
                torch.save(state_dict, f'{save_path}/{args.save_net}.pth')
                
            else:
                print(f'saved_models/{args.train_mode}/{str(args.set_seed)}/{args.save_net}.pth')
                if torch.cuda.device_count() > 1:
                    torch.save(net.module.state_dict(),
                               f'saved_models/{args.train_mode}/{str(args.set_seed)}/{args.save_net}.pth')
                else:
                    torch.save(net.state_dict(),
                               f'saved_models/{args.train_mode}/{str(args.set_seed)}/{args.save_net}.pth')
            best_acc = test_acc
            best_epoch = epoch

        if args.train_mode == 'adv' and epoch % 5 == 0:
            adv_acc, predicted = test_on_adv(args, net, testloader, device)
            print(f'EPOCH:{epoch}, Adv acc: {adv_acc}')

else:
    net.load_state_dict(torch.load(args.load_net))
    

if args.load_net is None and args.active_log:
                wandb.log({'best_epoch': epoch ,'best_test_accuracy': best_acc
                    })
# test_acc, predicted = test(args, net, testloader, device)
# print(test_acc)
end = time.time()
simple_lapsed_time("Time taken to train/load the model", end-start)

if not args.plot_animation:
    start = time.time()
    if args.imgs is None:
        #images, labels = get_random_images(trainloader.dataset)
        images, labels, image_ids = get_random_images(testloader.dataset)
    elif -1 in args.imgs:
        dummy_imgs, _ = get_random_images(testloader.dataset)
        images, labels = get_noisy_images(torch.stack(dummy_imgs), testloader.dataset, net, device)
    elif -10 in args.imgs:
        image_ids = args.imgs[0]
        images = [testloader.dataset[image_ids][0]]
        labels = [testloader.dataset[image_ids][1]]
        for i in list(range(2)):
            temp = torch.zeros_like(images[0])
            if i == 0:
                temp[0,0,0] = 1
            else:
                temp[0,-1,-1] = 1

            images.append(temp)
            labels.append(0)

    elif -100 in args.imgs:
        torch.manual_seed(args.set_data_seed)
        transform_train = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        temp_trainset = torchvision.datasets.MNIST(
            root='~/data', train=True, download=True, transform=transform_train)
        trainloader_2 = torch.utils.data.DataLoader(
            temp_trainset, batch_size=128, shuffle=False, num_workers=2)
        # import ipdb; ipdb.set_trace()
        # image_ids = args.imgs[:-1]
        np.random.seed(args.set_seed)
        image_ids = np.random.choice(range(50000), 2)
        images = [trainloader.dataset[i][0] for i in image_ids]
        labels = [trainloader.dataset[i][1] for i in image_ids]
        l = np.random.choice(range(60000), 1)[0]
        images.append(trainloader_2.dataset[l][0])
        labels.append(trainloader_2.dataset[l][1])
        print(labels,l)

    else:
        # import ipdb; ipdb.set_trace()
        image_ids = args.imgs
        images = [trainloader.dataset[i][0] for i in image_ids]
        labels = [trainloader.dataset[i][1] for i in image_ids]
        print(labels)
    if args.adv:
        adv_net = AttackPGD(net, trainloader.dataset)
        if args.targeted:
            base_img = images[0]
            base_label = labels[0]
            images = [base_img, base_img]
            labels = [(labels[0] + 1) % 10, (labels[0] + 2) % 10]
        adv_preds, imgs = adv_net(torch.stack(images).to(device), torch.tensor(labels).to(device), targeted=args.targeted)
        images = [img.cpu() for img in imgs]
        if args.targeted:
            images = [base_img] + images
            labels = [base_label] + labels

    if args.noise_type:
        if args.noise_type == 'gaussian':
            print('In gaussian')
            base_img = images[0]
            base_label = labels[0]
            np.random.seed(0)
            noise1 = torch.from_numpy(np.float32(np.clip(
                np.random.normal(size=(3, 32, 32), scale=0.5), -1, 1)))
            noise2 = torch.from_numpy(np.float32(np.clip(
                np.random.normal(size=(3, 32, 32), scale=0.25), -0.5, 0.5)))
            images = [base_img,base_img+noise1, base_img+noise2 ]
            labels = [base_label,base_label,base_label]
        elif args.noise_type == 'rotation':
            base_img = images[0]
            base_label = labels[0]
            images = [base_img,torch.rot90(base_img, 1, [1, 2]), torch.rot90(base_img, 2, [1, 2]) ]
            labels = [base_label,base_label,base_label]
        elif args.noise_type == 'uniform_random':
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
            from PIL import Image
            np.random.seed(args.set_seed)
            im1 = transform_train(Image.fromarray(np.uint8(np.random.uniform(0,1,(32,32,3))*255)))
            im2 = transform_train(Image.fromarray(np.uint8(np.random.uniform(0,1,(32,32,3))*255)))
            im3 = transform_train(Image.fromarray(np.uint8(np.random.uniform(0,1,(32,32,3))*255)))
            # import ipdb; ipdb.set_trace()
            images = [im1,im2,im3]
            labels = [0,0,0]
        elif args.noise_type == 'random_shuffle':
            import ipdb; ipdb.set_trace()
            np.random.seed(args.set_seed)
            image_ids = np.random.choice(range(50000), 3)
            images = [trainloader.dataset[i][0] for i in image_ids]
            labels = [trainloader.dataset[i][1] for i in image_ids]
            print(image_ids,labels)
            c,h,w = images[0].shape
            # import ipdb; ipdb.set_trace()
            shuffle1 = images[0].clone().reshape(c,-1)[:,torch.randperm(h*w)].reshape(c,h,w)
            shuffle2 = images[1].clone().reshape(c,-1)[:,torch.randperm(h*w)].reshape(c,h,w)
            shuffle3 = images[2].clone().reshape(c,-1)[:,torch.randperm(h*w)].reshape(c,h,w)

            images = [shuffle1,shuffle2,shuffle3]
        elif args.noise_type == 'two_random_shuffle':
            np.random.seed(args.set_seed)
            image_ids = np.random.choice(range(50000), 3)
            images = [trainloader.dataset[i][0] for i in image_ids]
            labels = [trainloader.dataset[i][1] for i in image_ids]
            print(image_ids,labels)
            c,h,w = images[0].shape
            # import ipdb; ipdb.set_trace()
            shuffle2 = images[1].clone().reshape(c,-1)[:,torch.randperm(h*w)].reshape(c,h,w)
            shuffle3 = images[2].clone().reshape(c,-1)[:,torch.randperm(h*w)].reshape(c,h,w)

            images = [images[0],shuffle2,shuffle3]
            # labels = [base_label,base_label,base_label]
    # image_ids = args.imgs
    sampleids = '_'.join(list(map(str,image_ids)))
    # sampleids = '_'.join(list(map(str,labels)))
    planeloader = make_planeloader(images, args)
    preds = decision_boundary(args, net, planeloader, device)
    from utils import produce_plot_alt,produce_plot_x,produce_plot_sepleg

    net_name = args.net
    if args.net == 'WideResNet':
        net_name = f'WideResNet_{args.widen_factor}'
    os.makedirs(f'images/{net_name}/{args.train_mode}/{sampleids}/{str(args.set_seed)}', exist_ok=True)
    # plot_path = os.path.join('images', args.net, args.train_mode, sampleids,str(args.set_seed),'best')
    # args.plot_path = os.path.join('./images', args.net, args.train_mode, sampleids, args.extra_path)
    # plot_path = os.path.join(args.plot_path,sampleids,f'{net_name}_{args.set_seed}cifar10')
    # os.makedirs(f'{args.plot_path}/{sampleids}', exist_ok=True)
    plot_path = os.path.join(args.plot_path,f'{net_name}_{sampleids}_{args.set_seed}cifar10')
    os.makedirs(f'{args.plot_path}', exist_ok=True)
    produce_plot_sepleg(plot_path, preds, planeloader, images, labels, trainloader, title = 'best', temp=1.0,true_labels = None)
    produce_plot_alt(plot_path, preds, planeloader, images, labels, trainloader)

    # produce_plot_x(plot_path, preds, planeloader, images, labels, trainloader, title=title, temp=1.0,true_labels = None)
    end = time.time()
    simple_lapsed_time("Time taken to plot the image", end-start)

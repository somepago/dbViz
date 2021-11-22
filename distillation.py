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

from model import get_model, get_teacher_model
from data import get_data, make_planeloader
from utils import get_loss_function, get_scheduler, get_random_images, produce_plot, get_noisy_images, AttackPGD
from evaluation import train, test, test_on_trainset, decision_boundary, test_on_adv, train_og_distillation
from options import options
from utils import simple_lapsed_time



args = options().parse_args()
print(args)
#torch.manual_seed(args.set_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data/other training stuff
torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
criterion = get_loss_function(args)


if __name__ == '__main__':
    # First train teacher model?
    teacher_net = get_teacher_model(args, device)
    os.makedirs(f'saved_models/{args.train_mode}/', exist_ok=True)
    save_path = f'./saved_models/{args.train_mode}/'
    #f'./saved_models/distillation/{args.extra_path}' #sloppy hardcoding for now :)

    if args.opt == 'SGD':
        optimizer = optim.SGD(teacher_net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = get_scheduler(args, optimizer)

    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(teacher_net.parameters(), lr=args.lr)
    if args.teacher_loc == '':
        print('Training teacher network')
        for epoch in range(args.epochs):
            train_acc = train(args, teacher_net, trainloader, optimizer, criterion, device, args.train_mode)
            test_acc, predicted = test(args, teacher_net, testloader, device, epoch)
            print(f'EPOCH:{epoch}, Test acc: {test_acc}')
            if args.opt == 'SGD':
                scheduler.step()

            # Save checkpoint.
            if epoch == (args.epochs - 1):
                os.makedirs(save_path, exist_ok=True)
                if torch.cuda.device_count() > 1:
                    state_dict = teacher_net.module.state_dict()
                else:
                    state_dict = teacher_net.state_dict()
                torch.save(state_dict, f'{save_path}/{args.teacher_net.lower()}_{args.net.lower()}.pth')
            if args.dryrun:
                break

    if not args.only_teacher:
        if args.teacher_loc == '':
            args.teacher_loc = save_path
        # changing the code to load the teach from any location and any name name
        teacher_net.load_state_dict(torch.load(args.teacher_loc))
        #teacher_net.load_state_dict(torch.load(f'{args.teacher_loc}/{args.teacher_net.lower()}_{args.net.lower()}.pth'))
        student_net = get_model(args, device)
        if args.opt == 'SGD':
            optimizer = optim.SGD(student_net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = get_scheduler(args, optimizer)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(student_net.parameters(), lr=args.lr)

        criterion = nn.KLDivLoss()
        for epoch in range(args.epochs):
            #train_acc = train(args, teacher_net, trainloader, optimizer, criterion, device, args.train_mode)
            train_acc = train_og_distillation(args, student_net, teacher_net, trainloader, optimizer, criterion, device)
            test_acc, predicted = test(args, student_net, testloader, device, epoch)
            print(f'EPOCH:{epoch}, Test acc: {test_acc}')
            if args.opt == 'SGD':
                scheduler.step()

            # Save checkpoint.
            if epoch == (args.epochs - 1):
                os.makedirs(save_path, exist_ok=True)
                if torch.cuda.device_count() > 1:
                    state_dict = student_net.module.state_dict()
                else:
                    state_dict = student_net.state_dict()
                torch.save(state_dict, f'{save_path}/{args.save_net}.pth')
            if args.dryrun:
                break



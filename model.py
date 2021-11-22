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

from PIL import Image, ImageFilter

from models import *
#from utils import progress_bar

print('==> Building model..')
def get_model(args, device):
    if args.net in ['ResNet','resnet']:
        net = ResNet18()
    elif args.net in ['VGG','vgg']:
        net = VGG('VGG19')
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net in ['DenseNet','densenet']:
        net = DenseNet121()
    elif args.net == 'MobileNet':
        net = MobileNetV2()
    elif args.net == 'LeNet':
        net = LeNet()
    elif args.net in ['FCNet','fcnet']:
        net = FCNet()
    elif args.net in ['ViT4','vit']:
        net = ViT4()
    elif args.net == 'ViT_pt_interpolate':
        net = ViT_pt_interpolate()
    elif args.net == 'ViT_npt_interpolate':
        net = ViT_pt_interpolate(pretrained=False)
    elif args.net == 'ViT_pt':
    # from https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
        import timm
        net = timm.create_model("vit_small_patch16_224", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net == 'MLPMixer4':
        net = MLPMixer4()
    elif args.net == 'MLPMixer_pt':
        import timm
        net = timm.create_model("mixer_s16_224", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net == 'WideResNet':
        net = WideResNet(depth=28, num_classes=10, widen_factor=args.widen_factor)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)
    return net

def get_teacher_model(args, device):
    if args.teacher_net == 'ResNet':
        net = ResNet18()
    elif args.teacher_net == 'VGG':
        net = VGG('VGG19')
    elif args.teacher_net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.teacher_net == 'DenseNet':
        net = DenseNet121()
    elif args.teacher_net == 'MobileNet':
        net = MobileNetV2()
    elif args.teacher_net == 'LeNet':
        net = LeNet()
    elif args.teacher_net == 'FCNet':
        net = FCNet()
    elif args.teacher_net == 'ViT4':
        net = ViT4()
    elif args.teacher_net == 'ViT_pt_interpolate':
        net = ViT_pt_interpolate()
    elif args.teacher_net == 'ViT_pt':
    # from https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
        import timm
        net = timm.create_model("vit_small_patch16_224", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.teacher_net == 'MLPMixer4':
        net = MLPMixer4()
    elif args.teacher_net == 'WideResNet':
        net = WideResNet(depth=28, num_classes=10, widen_factor=args.widen_factor)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)
    return net

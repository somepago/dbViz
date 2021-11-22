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
from os.path import exists

args = options().parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)

### I am going to run this for only one model # model is in args.net
net = get_model(args, device)
net.load_state_dict(torch.load(args.model_path))

save_path = args.model_path
save_path = save_path.replace("/", "_").split(".")[0]
print(save_path)
print(args.save_epoch)

for run in range(args.epochs):
    random.seed(a=(args.set_data_seed + run), version=2)
    images, labels, image_ids = get_random_images(testloader.dataset)
    planeloader = make_planeloader(images, args)
    if run in args.save_epoch:
        print(run)
        preds = decision_boundary(args, net, planeloader, device)
        preds = torch.stack(preds).argmax(1).cpu()
        torch.save(preds, f"./predictions/{save_path}_{run}_preds.pth")

        pred_1 = torch.load(f"./predictions/{save_path}_{run}_preds.pth")
        print(torch.equal(preds, pred_1))



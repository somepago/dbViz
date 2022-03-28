import torch
import numpy as np
import os
from db.data import make_planeloader
from db.utils import produce_plot_alt, mixup_data, rand_bbox
from torch.autograd import Variable
import copy
# from model import get_teacher_model


# Training

def train(args, net, trainloader, optimizer, criterion, device, mode):
    acc = eval(f"train_{mode}")(args, net, trainloader, optimizer, criterion, device)
    return acc

def train_naive(args, net, trainloader, optimizer, criterion, device):
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
        if 'kl' in args.criterion:
            _, targets = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100.*correct/total

def train_mixup(args, net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = 1.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, True)
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
        if args.dryrun:
            break
    return 100. * correct / total

def train_cutmix(args, net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = 1.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        orig_inputs = inputs
        #### cutmix ###
        r = np.random.rand(1)
        if args.cutmix_beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = net(inputs)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            del output
            del inputs
            del target_a
            del target_b
        else:
            # compute output
            output = net(inputs)
            loss = criterion(output, targets)

        ### cutmix ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        orig_output = net(orig_inputs)
        train_loss += loss.item()
        _, predicted = orig_output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100. * correct / total

def train_soft_distillation(args, net, trainloader, optimizer, criterion, device):

    teacher = get_teacher_model(args, device)
    if torch.cuda.device_count() > 1:
        teacher.module.load_state_dict(torch.load(args.teacher_loc))
    else:
        teacher.load_state_dict(torch.load(args.teacher_loc))

    teacher.eval()

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        teacher_labels = teacher(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, teacher_labels) #make sure l2 loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100. * correct / total

def train_hard_distillation(args, net, trainloader, optimizer, criterion, device):

    teacher = copy.deepcopy(net)
    teacher.load_state_dict(torch.load(args.teacher_loc))
    teacher.eval()

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        teacher_labels = teacher(inputs)
        max_vals, max_indices = torch.max(teacher_labels, 1)
        teacher_labels = max_indices

        outputs = net(inputs)
        loss = criterion(outputs, teacher_labels) #make sure crossentropy
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100. * correct / total



def test(args, net, testloader, device, epoch,images=None,labels=None,planeloader=None):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.dryrun:
                break
    if args.plot_animation:
        if epoch%2 == 0:
            preds = decision_boundary(args, net, planeloader, device)
            plot_path  =  os.path.join(args.plot_path , f"{epoch:03d}" )
            produce_plot_alt(plot_path, preds, planeloader, images, labels, testloader, epoch)
    return 100.*correct/total, predicted_labels

def test_on_trainset(args, net, clean_trainloader, device):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(clean_trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.dryrun:
                break
    return 100.*correct/total, predicted_labels

def decision_boundary(args, net, loader, device):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            for output in outputs:
                predicted_labels.append(output)
    return predicted_labels

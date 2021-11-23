import torch
import torch.nn.functional as F
import numpy as np
import os
from data import make_planeloader
from utils import produce_plot_alt, mixup_data, rand_bbox, AttackPGD
from torch.autograd import Variable
import copy
from model import get_teacher_model


# Training

def train(args, net, trainloader, optimizer, criterion, device, mode, sam_radius=False):
    if mode == "naive":
        acc = eval(f"train_{mode}")(args, net, trainloader, optimizer, criterion, device, sam_radius=False)
    else:
        acc = eval(f"train_{mode}")(args, net, trainloader, optimizer, criterion, device)
    return acc

def train_naive(args, net, trainloader, optimizer, criterion, device, sam_radius=False):
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
        if sam_radius:
            with torch.no_grad():
                norm_factor = 0
                perturb_dict = {}
                for name, para in net.named_parameters():
                    norm_factor += (para.grad ** 2).sum()
                for name, para in net.named_parameters():
                    para.data += para.grad / norm_factor * sam_radius
                    perturb_dict[name] = para.grad / norm_factor * sam_radius
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

def train_og_distillation(args, student, teacher, trainloader, optimizer, criterion, device):
    #teacher = copy.deepcopy(net)
    #teacher.load_state_dict(torch.load(args.teacher_loc))
    teacher.eval()
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        teacher_outputs = teacher(inputs)
        #max_vals, max_indices = torch.max(teacher_labels, 1)
        #teacher_labels = max_indices

        outputs = student(inputs)
        loss = args.distill_temp*args.distill_temp*criterion(F.log_softmax(outputs/args.distill_temp, dim=1),F.softmax(teacher_outputs/args.distill_temp, dim=1)) # make sure KL_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.dryrun:
            break
    return 100. * correct / total

def train_adv(args, net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        #loss = criterion(outputs, targets)
        #loss.backward()
        #optimizer.step()


        _, predicted = outputs.max(1)
        if 'kl' in args.criterion:
            _, targets = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        del outputs

        adv_net = AttackPGD(net, trainloader.dataset, numsteps=10)
        _, adv_imgs = adv_net(inputs, targets)

        outputs = net(adv_imgs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if args.dryrun:
            break
    return 100.*correct/total

def train_adv_mix(args, net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        _, predicted = outputs.max(1)
        if 'kl' in args.criterion:
            _, targets = targets.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        del outputs

        adv_net = AttackPGD(net, trainloader.dataset, numsteps=10)
        _, adv_imgs = adv_net(inputs, targets)

        outputs = net(adv_imgs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if args.dryrun:
            break
    return 100.*correct/total

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

def test_on_adv(args, net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            adv_net = AttackPGD(net, testloader.dataset)
            _, inputs_adv = adv_net(inputs, targets)

            outputs = net(inputs_adv)
            _, predicted = outputs.max(1)
            predicted_labels.append(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.dryrun:
                break
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
            if args.dryrun:
                break
    return predicted_labels

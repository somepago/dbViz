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


torch.manual_seed(args.set_data_seed)
trainloader, testloader = get_data(args)
torch.manual_seed(args.set_seed)
test_accs = []
train_accs = []
net = get_model(args, device)
ious = []

net_dict = dict([('googlenet', 'GoogLeNet'), ('vit','ViT4'), ('resnet','ResNet'), ('densenet','DenseNet'), ('vgg','VGG'), ('fcnet','FCNet'), ('wideresnet', 'WideResNet')])


def calculate_iou_simple(pred_arr1, pred_arr2):
    diff = pred_arr1.shape[0] - (pred_arr1 - pred_arr2).count_nonzero()
    iou = diff / pred_arr1.shape[0]
    return iou.cpu()

def get_path_pairs(paths):
    path_pairs = []
    for index1 in range(len(paths)):
        for index2 in range(index1, len(paths)):
            if index1 == index2:
                for i in range(len(paths[index1])):
                    for j in range(i+1, len(paths[index2])):
                        path_pairs.append((paths[index1][i], paths[index1][j], index1, index2))
            else:
                for i in range(len(paths[index1])):
                    for j in range(len(paths[index2])):
                        path_pairs.append((paths[index1][i], paths[index1][j], index1, index2))
    return path_pairs

def get_distill_paths(base_dir):
    clean_nets = {}
    student_nets = {}
    temp_paths = os.listdir(base_dir)
    teachers = []
    students = []
    for temp_path in sorted(temp_paths):
        if 'just' in temp_path:
            for temp_path2 in os.listdir(os.path.join(base_dir, temp_path)):
                clean_nets[f"{temp_path2.split('_')[0]}_seed{temp_path.split('seed')[-1]}"] = os.path.join(base_dir, temp_path, temp_path2)
        elif 'to' in temp_path: 
            teacher = temp_path.split('_to_')[0]
            if teacher not in teachers: 
                teachers.append(teacher)
            student = temp_path.split('_to_')[-1].split('_')[0]
            if student not in students:
                students.append(student)
            for temp_path2 in sorted(os.listdir(os.path.join(base_dir, temp_path))):
                if f"{student}_seed{temp_path.split('seed')[-1]}" not in student_nets:
                    student_nets[f"{student}_seed{temp_path.split('seed')[-1]}"] = {}
                if teacher in temp_path2:
                    student_nets[f"{student}_seed{temp_path.split('seed')[-1]}"]['teacher'] = os.path.join(base_dir, temp_path, temp_path2)
                elif student in temp_path2:
                    student_nets[f"{student}_seed{temp_path.split('seed')[-1]}"]['student'] = os.path.join(base_dir, temp_path, temp_path2)
                

    return clean_nets, student_nets, students, teachers

def get_net_name(path):
    temp_name = path.split('/')[-1].split('_')[0]
    return net_dict[temp_name]
                
def get_total_paths(base_path):
    temp_paths = sorted(os.listdir(base_path))
    path_list = []
    for temp_path in temp_paths:
        new_paths = sorted(os.listdir(os.path.join(base_path, temp_path)))
        new_paths = [os.path.join(base_path, temp_path, new_path) for new_path in new_paths]
        path_list.append(new_paths)
    return path_list

clean_nets, student_nets, students, teachers  = get_distill_paths(args.load_net)

clean_pred_dict = {} 
for (k, v) in clean_nets.items(): 
    args.net = get_net_name(v)
    if args.net == 'WideResNet':
        args.widen_factor = int(path.split('/')[-1].split('_')[1])
    net = get_model(args, device)
    net.load_state_dict(torch.load(v))
    pred_arr = []
    for run in range(args.epochs):
        random.seed(a=(args.set_data_seed+run), version=2)
        images, labels, image_ids = get_random_images(testloader.dataset)
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        pred_arr.append(torch.stack(preds).argmax(1).cpu())
    clean_pred_dict[k] = torch.cat(pred_arr)

student_pred_dict = {} 
for k in student_nets.keys(): 
    temp_dict = student_nets[k]
    net_name = temp_dict['student'].split('/')[-1].split('.pth')[0]
    v = temp_dict['student']
    args.net = get_net_name(net_name)
    if args.net == 'WideResNet':
        args.widen_factor = int(path.split('/')[-1].split('_')[1])
    net = get_model(args, device)
    net.load_state_dict(torch.load(v))
    pred_arr = []
    for run in range(args.epochs):
        random.seed(a=(args.set_data_seed+run), version=2)
        images, labels, image_ids = get_random_images(testloader.dataset)
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        pred_arr.append(torch.stack(preds).argmax(1).cpu())
    student_pred_dict[k] = torch.cat(pred_arr)

teacher_pred_dict = {} 
for k in student_nets.keys(): 
    temp_dict = student_nets[k]
    net_name = temp_dict['teacher'].split('/')[-1].split('.pth')[0]
    v = temp_dict['teacher']
    args.net = get_net_name(net_name)
    if args.net == 'WideResNet':
        args.widen_factor = int(path.split('/')[-1].split('_')[1])
    net = get_model(args, device)
    net.load_state_dict(torch.load(v))
    pred_arr = []
    for run in range(args.epochs):
        random.seed(a=(args.set_data_seed+run), version=2)
        images, labels, image_ids = get_random_images(testloader.dataset)
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        pred_arr.append(torch.stack(preds).argmax(1).cpu())
    teacher_pred_dict[k] = torch.cat(pred_arr)

iou_mat = torch.zeros(2, len(students))
tot = torch.zeros_like(iou_mat)

for k in teacher_pred_dict.keys():
    if k in teacher_pred_dict and k in student_pred_dict and k in clean_pred_dict:
        teacher_preds = teacher_pred_dict[k]
        student_preds = student_pred_dict[k]
        clean_preds = clean_pred_dict[k]
        for i, student_name in enumerate(students):
            if student_name in k:
                iou_mat[0,i] += calculate_iou_simple(teacher_preds, student_preds)
                iou_mat[1,i] += calculate_iou_simple(teacher_preds, clean_preds)
                tot[0,i] += 1
                tot[1,i] += 1
                break


print(iou_mat)
print(tot)
print(students)


'''

iou_mat = torch.zeros((len(paths), len(paths)))
iou_count = torch.zeros((len(paths), len(paths)))
path_pairs = get_path_pairs(paths)

    
print(f'Calculating IOU for {sum([len(path) for path in paths])} networks')
predictions = [[] for _ in range(len(paths))]
start = time.time()
for i, group in enumerate(paths):
    for j, path in enumerate(group):
        start = time.time()
        args.net = get_net_name(path)
        if args.net == 'WideResNet':
            args.widen_factor = int(path.split('/')[-1].split('_')[1])
        net = get_model(args, device)
        net.load_state_dict(torch.load(path))
        pred_arr = []
        for run in range(args.epochs):
            random.seed(a=(args.set_data_seed+run), version=2)
            images, labels, image_ids = get_random_images(testloader.dataset)
            planeloader = make_planeloader(images, args)
            preds = decision_boundary(args, net, planeloader, device)
            pred_arr.append(torch.stack(preds).argmax(1).cpu())
        predictions[i].append(torch.cat(pred_arr)) 
print(f'Done calculating IOU for {sum([len(path) for path in paths])} networks')
end = time.time()
simple_lapsed_time(f"Average time for {sum([len(path) for path in paths])} networks, and {args.epochs} runs :", (end - start) / sum([len(path) for path in paths])/args.epochs)
simple_lapsed_time(f'Total time:', (end - start))



start = time.time()
total = 0
running_total = 0
for i in range(len(predictions)):
    for j in range(i, len(predictions)):
        if i == j:
            for k in range(len(predictions[i])):
                for l in range(k+1, len(predictions[j])):
                    iou = calculate_iou_simple(predictions[i][k].to(device), predictions[j][l].to(device))
                    iou_mat[i, j] += iou
                    iou_count[i, j] += 1
                    total += 1
                    end = time.time()
                    running_total += end-start
                    running_avg = running_total / total
                    if total % 10 == 0:
                        print(iou_mat / iou_count)
                        simple_lapsed_time("Estimated time remaining:", running_avg * len(path_pairs) - running_total)
                        
        else:
            for k in range(len(predictions[i])):
                for l in range(len(predictions[j])):
                    iou = calculate_iou_simple(predictions[i][k].to(device), predictions[j][l].to(device))
                    iou_mat[i, j] += iou
                    iou_count[i, j] += 1
                    total += 1
                    end = time.time()
                    running_total += end-start
                    running_avg = running_total / total
                    if total % 10 == 0:
                        print(iou_mat / iou_count)
                        simple_lapsed_time("Estimated time remaining:", running_avg * len(path_pairs) - running_total)
            
print(iou_mat / iou_count)
                    
            
'''

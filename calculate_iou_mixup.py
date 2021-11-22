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
from os.path import exists

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


def calculate_iou_simple(pred_arr1, pred_arr2):
    print(pred_arr1.shape, pred_arr2.shape)
    print((pred_arr1 - pred_arr2).count_nonzero())
    diff = pred_arr1.shape[0] - (pred_arr1 - pred_arr2).count_nonzero()
    iou = diff / pred_arr1.shape[0]
    return iou.cpu()


def get_paths(students, teachers):
    students_paths = {}
    teacher_paths = {}

    t_p = "./saved_models/naive"
    s_p = "./saved_models/mixup"
    print(t_p)
    print(s_p)
    for teacher in teachers:
        teacher_paths[teacher] = []
        students_paths[teacher] = []
        for seed in [1, 2, 3]:
            if teacher == "ViT_pt_interpolate":
                path = f"{t_p}/1/{teacher}_naive_1_cifar10.pth"
                teacher_paths[teacher].append(path)

                path = f"{s_p}/1/{teacher}_mixup_1_cifar10.pth"
                students_paths[teacher].append(path)
                break

            else:
                path = f"{t_p}/{seed}/{teacher}_naive_{seed}_cifar10.pth"
                teacher_paths[teacher].append(path)

                path = f"{s_p}/{seed}/{teacher}_mixup_{seed}_cifar10.pth"
                students_paths[teacher].append(path)

            print(path)

    return students_paths, teacher_paths




students = args.student_lists #["ResNet", "WideResNet", "DenseNet", "VGG"]
teachers = args.teacher_lists #["ResNet", "WideResNet", "DenseNet", "VGG"]

student_paths, teacher_paths = get_paths(students, teachers)

teacher_bounds = {}
student_bounds = {}

print("teachers")
for teacher in teacher_paths.keys():
    models = teacher_paths[teacher]

    all_seeds = []
    for model_seed in models:
        args.net = teacher
        print(model_seed)
        net = get_model(args, device)
        net.load_state_dict(torch.load(model_seed))
        net.eval()

        pred_arr = []
        for run in range(args.epochs):
            print(run)
            random.seed(a=(args.set_data_seed + run), version=2)
            images, labels, image_ids = get_random_images(testloader.dataset)
            planeloader = make_planeloader(images, args)
            preds = decision_boundary(args, net, planeloader, device)
            pred_arr.append(torch.stack(preds).argmax(1).cpu())

        all_seeds.append(torch.stack(pred_arr).view(-1))

    teacher_bounds[teacher] = torch.stack(all_seeds).reshape(-1)


print("students")
for student in student_paths.keys():
    models = student_paths[student]

    all_seeds = []
    for model_seed in models:
        args.net = student
        print(model_seed)
        net = get_model(args, device)
        net.load_state_dict(torch.load(model_seed))
        net.eval()

        pred_arr = []
        for run in range(args.epochs):
            print(run)
            random.seed(a=(args.set_data_seed + run), version=2)
            images, labels, image_ids = get_random_images(testloader.dataset)
            planeloader = make_planeloader(images, args)
            preds = decision_boundary(args, net, planeloader, device)
            pred_arr.append(torch.stack(preds).argmax(1).cpu())

        all_seeds.append(torch.stack(pred_arr).view(-1))

    student_bounds[student] = torch.stack(all_seeds).reshape(-1)




for teacher in teacher_paths.keys():
    teacher_db = teacher_bounds[teacher]
    student_db = student_bounds[teacher]

    iou_mat = calculate_iou_simple(teacher_db, student_db)
    print(teacher, iou_mat)


print(iou_mat)



# iou_mat = torch.zeros(2, len(students))
# tot = torch.zeros_like(iou_mat)

# for k in teacher_pred_dict.keys():
#     if k in teacher_pred_dict and k in student_pred_dict and k in clean_pred_dict:
#         teacher_preds = teacher_pred_dict[k]
#         student_preds = student_pred_dict[k]
#         clean_preds = clean_pred_dict[k]
#         for i, student_name in enumerate(students):
#             if student_name in k:
#                 iou_mat[0,i] += calculate_iou_simple(teacher_preds, student_preds)
#                 iou_mat[1,i] += calculate_iou_simple(teacher_preds, clean_preds)
#                 tot[0,i] += 1
#                 tot[1,i] += 1
#                 break
#
#
# print(iou_mat)
# print(tot)
# print(students)


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

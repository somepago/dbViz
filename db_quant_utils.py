import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from PIL import Image
import os

from data import make_planeloader
from evaluation import decision_boundary

##Margin stuff

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


## Class Skips

def rel_index(class_index,trainset,testset,noise_rate):
    l_all_train = np.where(np.array(trainset.targets) == class_index)[0]
    l_mis_cls = 0
    l_fromcls_mislab = 0
    l_corr_cls = l_all_train
    if noise_rate > 0:
        l_mis = np.where(np.array(trainset.targets) != np.array(trainset.true_targets))[0] #mislabeled images
        l_corr = np.where(np.array(trainset.targets) == np.array(trainset.true_targets))[0] #correctly labeled images
        l_mis_cls = np.intersect1d(l_all_train,l_mis)
        l_corr_cls = np.intersect1d(l_all_train,l_corr)
        l_fromcls_mislab = np.intersect1d(np.where(np.array(trainset.true_targets) == class_index)[0],l_mis)
    l_all_test = np.where(np.array(testset.targets) == class_index)[0]

    return l_all_train, l_mis_cls, l_corr_cls,l_all_test,l_fromcls_mislab


def count_skips(dlist1,dlist2,loader1,loader2,num_steps,net,device):
    skips_list = []
    for i in range(len(dlist1)):
        img1 = loader1.dataset[dlist1[i]][0].unsqueeze(0).expand(num_steps,-1,-1,-1)
        img2 = loader2.dataset[dlist2[i]][0].unsqueeze(0).expand(num_steps,-1,-1,-1)
        img1 = img1.to(device)
        img2 = img2.to(device)
        alphas = torch.from_numpy(np.linspace(0,1,num_steps)).reshape(num_steps,1,1,1)
        alphas  = alphas.to(device)
        img_batch = img1 + alphas*(img2-img1)
        img_batch = img_batch.to(device=device, dtype=torch.float)
        preds = torch.argmax(net(img_batch),dim=1).cpu().numpy()
        skip = (np.diff(preds)!=0).sum()
        skips_list.append(skip)
    return skips_list



def connected_components(preds,args):
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds)
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    resolution = int(args.resolution)
    img = np.zeros((resolution, resolution)).astype(np.uint8)
    img[np.arange(resolution).repeat(resolution), np.tile(np.arange(resolution), resolution)] = class_pred
    unique_classes =  np.unique(img)

    cc_counts = []
    for lbl in unique_classes:
        _, count = measure.label((img==lbl).astype(np.uint8), background=0, return_num=True, connectivity=2)
        cc_counts.append(count)
    
    if args.plot_path is not None:
        path = args.plot_path
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        colors = np.array([[  3,  41,   8],
                            [231,  50, 144],
                            [144,  96,  19],
                            [141,  21, 179],
                            [ 26,  42, 130],
                            [215,  93,   7],
                            [ 85,  88, 251],
                            [137, 112, 156],
                            [167, 245, 192],
                            [243,  20, 230]])
        Image.fromarray(colors[img].astype(np.uint8)).save(f'{path}_fragmentation.png')
    return np.sum(cc_counts)

import wandb
def num_connected_components(dlist1,dlist2, loader1,loader2, num_samples,net,device,args):
    cc_list = []
    for i in range(num_samples):
        # import ipdb; ipdb.set_trace()
        dirlist = np.random.choice(dlist1, 2)
        dirlist2 = np.random.choice(dlist2, 1)
        images = [loader1.dataset[j][0] for j in dirlist]
        images.append(loader2.dataset[dirlist2[0]][0])
        planeloader = make_planeloader(images, args)
        preds = decision_boundary(args, net, planeloader, device)
        ncc = connected_components(preds,args)
        cc_list.append(ncc)
        # print(dirlist,dirlist2,ncc)
        if i%100==0:
            if args.active_log:
                wandb.log({'iteration':i})
    return cc_list
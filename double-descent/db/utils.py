'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



def get_loss_function(args):
    if args.criterion == '':
        criterion = nn.CrossEntropyLoss()
    elif 'kl' in args.criterion:
        def kl_loss(outputs, targets):
            return torch.nn.functional.kl_div(F.log_softmax(outputs, dim=1),F.softmax(targets, dim=1))
        criterion = kl_loss
    elif 'MSE' in args.criterion:
        criterion = torch.nn.MSELoss()
    elif 'mixup' in args.criterion:
        criterion = mixup_criterion

    return criterion


def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler


def get_random_images(trainset):
    imgs = []
    labels = []
    ids = []
    while len(imgs) < 3:
        idx = random.randint(0, len(trainset)-1)
        img, label = trainset[idx]
        if label not in labels:
            imgs.append(img)
            labels.append(label)
            ids.append(idx)

    return imgs, labels, ids

def get_noisy_images(dummy_imgs, dataset, net, device):
    #dm = torch.tensor(dataset.transform.transforms[-1].mean)[:, None, None]
    #ds = torch.tensor(dataset.transform.transforms[-1].std)[:, None, None]
    dm = (0.5 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
    ds = (0.25 * torch.ones(dummy_imgs.shape[0])).unsqueeze(-1).unsqueeze(-1)
    #imgs = torch.rand(dummy_imgs.shape)
    #imgs = (imgs - dm) / ds
    #imgs = imgs.to(device)
    new_imgs = []
    new_labels = []
    net.eval()
    with torch.no_grad():
        while len(new_labels) < dummy_imgs.shape[0]:
            imgs = torch.rand(dummy_imgs.shape)
            imgs = (imgs - dm) / ds
            imgs = imgs.to(device)
            outputs = net(imgs)
            _, labels = outputs.max(1)
            for i, label in enumerate(labels):
                if label.cpu() not in new_labels:
                    new_imgs.append(imgs[i].cpu())
                    new_labels.append(label.cpu())
    #new_imgs = [img.cpu() for img in imgs]
    #new_labels = [label.cpu() for label in labels]
    return new_imgs, new_labels


def _get_class_preds(planeset, preds, label, avoid_labels=None):
    x = []
    y = []
    vals = []
    for i, pred in enumerate(preds):
        val = torch.softmax(pred,0).max()
        class_pred = pred.argmax()
        if avoid_labels is None:
            if class_pred == label:
                x.append(planeset.coefs1[i].cpu().numpy())
                y.append(planeset.coefs2[i].cpu().numpy())
                vals.append(val.cpu().numpy())
        else:
            if class_pred not in avoid_labels:
                x.append(planeset.coefs1[i].cpu().numpy())
                y.append(planeset.coefs2[i].cpu().numpy())
                vals.append(val.cpu().numpy())
    return vals, x, y


def imscatter(x, y, image, ax=None, zoom=1):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def produce_plot(path, preds, planeloader, images, labels, trainloader, method='greys'):
    color_list = ['Reds', 'Blues', 'Greens']
    other_colors = ['Purples', 'Oranges', 'YlOrBr', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    import ipdb; ipdb.set_trace()
    for i, label in enumerate(labels):
        vals, x, y = _get_class_preds(planeloader.dataset, preds, label, avoid_labels=None)
        ax1.scatter(x, y, c=vals, cmap=color_list[i], label=f'class={label}')
    if method=='greys':
        vals, x, y = _get_class_preds(planeloader.dataset, preds, label, avoid_labels=labels)
        ax1.scatter(x, y, c=vals, cmap='Greys', label=f'class=other')

    elif method=='all':
        indcs = set(list(range(list(preds[0].shape)[0]))) - set(labels)
        for i, ind in enumerate(indcs):
            if i not in labels:
                vals, x, y = _get_class_preds(planeloader.dataset, preds, ind, avoid_labels=None)
                ax1.scatter(x, y, c=vals, cmap=other_colors[i], label=f'class={i}')


    ax1.legend

    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    for i, image in enumerate(images):
        img = torch.clamp(image * ds + dm, 0, 1)
        img = img.cpu().numpy().transpose(1,2,0)
        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    red_patch = mpatches.Patch(color='red', label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color='blue', label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color='green', label=f'{classes[labels[2]]}')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    if path is not None:
        os.makedirs('images', exist_ok=True)
        plt.savefig(f'images/{path}.png')
    plt.close(fig)
    return

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if criterion == None:
        criterion = nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def produce_plot_alt(path, preds, planeloader, images, labels, trainloader, epoch='best', temp=1.0):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('tab10')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['airpl', 'autom', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1 = plt.subplots()
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
    ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    for i, image in enumerate(images):
        img = torch.clamp(image * ds + dm, 0, 1)
        img = img.cpu().numpy().transpose(1,2,0)
        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title(f'Epoch: {epoch}')
    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        #os.makedirs(path.split, exist_ok=True)
        plt.savefig(f'{path}.png',bbox_extra_artists=(legend1,), bbox_inches='tight')
    plt.close(fig)
    return

def produce_plot_x(path, preds, planeloader, images, labels, trainloader, title='best', temp=1.0,true_labels = None):
    import seaborn as sns
    sns.set_style("whitegrid")
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 15,}                  
    sns.set_context("paper", rc = paper_rc,font_scale=1.5)  
    plt.rc("font", family="Times New Roman")
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('tab10')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['AIRPL', 'AUTO', 'BIRD', 'CAT', 'DEER',
                   'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']

    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1  = plt.subplots()

    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    # legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
    # ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    # import ipdb; ipdb.set_trace()
    markerd = {
        0: 'o',
        1 : '^',
        2 : 'X'
    }
    for i, image in enumerate(images):
        coord = coords[i]
        plt.scatter(coord[0], coord[1], s=150, c='red', marker=markerd[i])
    red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
    if true_labels is not None:
        p0 = mpatches.Patch(color =cmaplist[true_labels[0]] , label=f'{classes[true_labels[0]]}')
        p1 = mpatches.Patch(color =cmaplist[true_labels[1]] , label=f'{classes[true_labels[1]]}')
        p2 = mpatches.Patch(color =cmaplist[true_labels[2]] , label=f'{classes[true_labels[2]]}')
        ph = mpatches.Patch(color = 'white' , label="Assigned:",visible=False)

        # import ipdb; ipdb.set_trace()
        leg2 = plt.legend(handles=[ph,red_patch, blue_patch, green_patch],  loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=4, fancybox=True, shadow=True,prop={'size': 14},handletextpad=0.2,columnspacing=0.4)
        ax1.add_artist(leg2)
        plt.legend(handles=[p0, p1, p2], loc='lower center', bbox_to_anchor=(0.5, -0.05),
                ncol=3, fancybox=True, shadow=True,prop={'size': 18},handletextpad=0.2,columnspacing=0.7,borderpad=0.2)
    else:
        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05),
                ncol=3, fancybox=True, shadow=True,prop={'size': 18},handletextpad=0.2,columnspacing=0.7,borderpad=0.2)
    plt.title(f'{title}',fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)    
    # fig.tight_layout()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1.2, bottom = 0, right = 1, left = 0, 
    #             hspace = 0, wspace = 0)
    plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        #os.makedirs(path.split, exist_ok=True)
        if true_labels is not None :
            plt.savefig(f'{path}_x.png',bbox_extra_artists=(leg2,), bbox_inches='tight')
        else:
            plt.savefig(f'{path}_x.png', bbox_inches='tight')

    plt.close(fig)
    return


def produce_plot_sepleg(path, preds, planeloader, images, labels, trainloader, title='best', temp=0.01,true_labels = None):
    import seaborn as sns
    sns.set_style("whitegrid")
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 15,}                  
    sns.set_context("paper", rc = paper_rc,font_scale=1.5)  
    plt.rc("font", family="Times New Roman")
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('gist_rainbow')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['AIRPL', 'AUTO', 'BIRD', 'CAT', 'DEER',
                   'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']
    cmaplist = [cmaplist[45],cmaplist[30],cmaplist[170],cmaplist[150],cmaplist[65],cmaplist[245],cmaplist[0],cmaplist[220],cmaplist[180],cmaplist[90]]
    cmaplist[2] = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0)
    cmaplist[4] = (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)

    # cmaplist = [cmaplist[int(25*j)] for j in range(len(classes))]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1  = plt.subplots()

    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=0.5, s=0.1)

    # scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    # legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
    # ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    # import ipdb; ipdb.set_trace()
    markerd = {
        0: 'o',
        1 : '^',
        2 : 'X'
    }
    for i, image in enumerate(images):
        coord = coords[i]
        plt.scatter(coord[0], coord[1], s=150, c='black', marker=markerd[i])
    # red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    # blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    # green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')

    labelinfo = {
        'labels' : [classes[i] for i in labels]
    }
    if true_labels is not None:
        labelinfo['true_labels'] = [classes[i] for i in true_labels] 


    # plt.title(f'{title}',fontsize=20)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)    
    # ax1.get_legend().remove()
    # fig.tight_layout()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1.2, bottom = 0, right = 1, left = 0, 
    #             hspace = 0, wspace = 0)
    plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        # pickle.dump( labelinfo, open(f"{path}_labelinfo.p", "wb" ) )
        #os.makedirs(path.split, exist_ok=True)
        plt.savefig(f'{path}_x.png', bbox_inches='tight')


    plt.close(fig)
    return

def produce_plot_catdog(path, preds, planeloader, images, labels, trainloader, epoch='best', temp=1.0):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    col_map = cm.get_cmap('tab10')
    cmaplist = [col_map(i) for i in range(col_map.N)]
    classes = ['cat', 'dog']

    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist, N=len(classes))
    fig, ax1 = plt.subplots()
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds / temp)
    val = torch.max(preds,dim=1)[0].cpu().numpy()
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    x = planeloader.dataset.coefs1.cpu().numpy()
    y = planeloader.dataset.coefs2.cpu().numpy()
    label_color_dict = dict(zip([*range(10)], cmaplist))

    color_idx = [label_color_dict[label] for label in class_pred]
    scatter = ax1.scatter(x, y, c=color_idx, alpha=val, s=0.1)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in label_color_dict.values()]
    legend1 = plt.legend(markers, classes, numpoints=1,bbox_to_anchor=(1.01, 1))
    ax1.add_artist(legend1)
    coords = planeloader.dataset.coords

    dm = torch.tensor(trainloader.dataset.transform.transforms[-1].mean)[:, None, None]
    ds = torch.tensor(trainloader.dataset.transform.transforms[-1].std)[:, None, None]
    for i, image in enumerate(images):
        img = torch.clamp(image * ds + dm, 0, 1)
        img = img.cpu().numpy().transpose(1,2,0)
        coord = coords[i]
        imscatter(coord[0], coord[1], img, ax1)

    red_patch = mpatches.Patch(color =cmaplist[labels[0]] , label=f'{classes[labels[0]]}')
    blue_patch = mpatches.Patch(color =cmaplist[labels[1]], label=f'{classes[labels[1]]}')
    green_patch = mpatches.Patch(color =cmaplist[labels[2]], label=f'{classes[labels[2]]}')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title(f'Epoch: {epoch}')
    if path is not None:
        img_dir = '/'.join([p for p in (path.split('/'))[:-1]])
        os.makedirs(img_dir, exist_ok=True)
        
        #os.makedirs(path.split, exist_ok=True)
        plt.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)
    return




def connected_components(preds,args,path=None):
    import torch.nn as nn
    preds = torch.stack((preds))
    preds = nn.Softmax(dim=1)(preds)
    class_pred = torch.argmax(preds, dim=1).cpu().numpy()
    resolution = int(args.resolution)
    img = np.zeros((resolution, resolution)).astype(np.uint8)
    img[np.arange(resolution).repeat(resolution), np.tile(np.arange(resolution), resolution)] = class_pred
    unique_classes =  np.unique(img)
    
    from skimage import measure
    from PIL import Image
    cc_counts = []
    for lbl in unique_classes:
        _, count = measure.label((img==lbl).astype(np.uint8), background=0, return_num=True, connectivity=2)
        cc_counts.append(count)
    
    

    if path is not None:
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
        Image.fromarray(colors[img].astype(np.uint8)).save(f'{path}_forcc_random.png')
    return np.sum(cc_counts)



# code to invert colors


# from PIL import Image
# import PIL.ImageOps    
# k = '64'

# path = '/cmlscratch/gowthami/deci_bounds/double-descent/imgs/train_paper/3random_nipyspec/'

# image = Image.open(path + f'_672_766_781_0.2_rn{k}_x.png')
# if image.mode == 'RGBA':
#     r,g,b,a = image.split()
#     rgb_image = Image.merge('RGB', (r,g,b))

#     inverted_image = PIL.ImageOps.invert(rgb_image)

#     r2,g2,b2 = inverted_image.split()

#     final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

#     final_transparent_image.save(path + f'{k}.png')

# else:
#     inverted_image = PIL.ImageOps.invert(image)
#     inverted_image.save(path + f'{k}.png')
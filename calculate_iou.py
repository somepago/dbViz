'''Calculate IOU'''
import torch
import random

from options import options
import os
import argparse


args = options().parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_iou_simple(pred_arr1, pred_arr2):
    diff = pred_arr1.shape[0] - (pred_arr1 - pred_arr2).count_nonzero()
    iou = diff / pred_arr1.shape[0]
    return iou.cpu()


'''
Here we load from a directory that has the following structure: 
args.load_net = /path/to/networks/

/path/to/networks/
    /net1/
        /predictions/
            /net1_preds1.pth
            /net1_preds2.pth
            /net1_preds3.pth

    /net2/
        /predictions/
            /net2_preds1.pth
            /net2_preds2.pth
            /net2_preds3.pth

    .
    .
    .
    
'''
 
# Get paths
groups = []
for p in sorted(os.listdir(args.load_net)):
    for q in sorted(os.listdir(os.path.join(args.load_net, p))):
        group_paths = []
        if 'prediction' in q:
            for s in sorted(os.listdir(os.path.join(args.load_net, p, q))):
                group_paths.append(os.path.join(args.load_net, p, q, s))
            groups.append(group_paths)

#groups = [[os.path.join(args.load_net, p, q) for q in sorted(os.listdir(os.path.join(args.load_net, p)))] for p in sorted(os.listdir(args.load_net)) if 'predictions' in p]
iou_mat = torch.zeros((len(groups), len(groups)))
total_mat = torch.zeros((len(groups), len(groups)))
# Now, iterate over the groups of paths
for i, g1 in enumerate(groups):
    for j, g2 in enumerate(groups): 
        for p1 in g1:
            for p2 in g2: 
                if p1 == p2:
                    continue # Skip comparison to the same exact path
                
                # Load the predictions
                pred1 = torch.load(p1)
                pred2 = torch.load(p2)
                pred1 = torch.cat(pred1)
                pred2 = torch.cat(pred2)

                iou_mat[i][j] += calculate_iou_simple(pred1, pred2)
                total_mat[i][j] += 1


print(iou_mat / total_mat)
                    

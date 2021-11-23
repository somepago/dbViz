"""Implement an ArgParser for main.py ."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    """
    parser = argparse.ArgumentParser(description='Argparser for main decision boundary code')
    # Training details
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='num epochs')
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--net', default='ResNet', type=str)
    parser.add_argument('--baseset', default='CIFAR10', type=str,
                            choices=['CIFAR10', 'CIFAR100','SVHN',
                            'CIFAR100_label_noise', 'CIFAR_load'])
    parser.add_argument('--load_net', type=str, default=None)
    parser.add_argument('--load_data', type=str, default=None)
    parser.add_argument('--save_net', type=str, default=None)
    parser.add_argument('--train_mode', type=str, default='naive')
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--sam_radius', type=float)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--bs', default=128 , type=int)

    # Changes to training?
    parser.add_argument('--cutmix_beta', default=1.0, type=float, help='hyperparameter beta for cutmix')
    parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')
    parser.add_argument('--teacher_loc', type=str, default='')
    parser.add_argument('--teacher_net', default='ResNet', type=str)
    parser.add_argument('--criterion', type=str, default='')
    parser.add_argument('--adv', action='store_true', help='Adversarially attack images?')
    parser.add_argument('--targeted', action='store_true', help='Targeted adversarial attacks?')
    parser.add_argument('--noise_type', default=None, type=str)
    parser.add_argument('--widen_factor', type=int, default=10)
    parser.add_argument('--distill_temp', type=float, default=30.0)
    parser.add_argument('--only_teacher', action='store_true')

    # Plotting details
    parser.add_argument('--resolution', default=500, type=float, help='resolution for plot')
    parser.add_argument('--range_l', default=0.1, type=float, help='how far `left` to go in the plot')
    parser.add_argument('--range_r', default=0.1, type=float, help='how far `right` to go in the plot')
    parser.add_argument('--temp', default=5.0, type=float)
    parser.add_argument('--plot_method', default='greys', type=str)
    parser.add_argument('--plot_train_imgs', action='store_true')
    parser.add_argument('--plot_animation', action='store_true')
    parser.add_argument('--plot_path', type=str, default=None)
    parser.add_argument('--extra_path', type=str, default=None)
    parser.add_argument('--imgs', default=None,
                            type=lambda s: [int(item) for item in s.split(',')], help='which images ids to plot')

    # Reproducibility?
    parser.add_argument('--active_log', action='store_true')
    parser.add_argument('--set_seed', default=1 , type=int)
    parser.add_argument('--set_data_seed', default=1 , type=int)
    parser.add_argument('--student_lists', nargs="+", default=["ResNet"], type=str, help='which models in students')
    parser.add_argument('--teacher_lists', nargs="+", default=["ResNet"], type=str, help='which models in teachers')

    # just save preds
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_path_2', type=str, default=None)
    parser.add_argument('--save_epoch',  nargs="+", type=int, default=None)

    return parser

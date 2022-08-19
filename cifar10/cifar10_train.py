# coding: utf-8

# Code adapted from the Confident Learning CIFAR-10 reproduce folder:
# https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_train_crossval.py
# Northcutt, C.; Jiang, L.; and Chuang, I. 2021. 
# Confident learning: Estimating uncertainty in dataset labels. 
# Journal of Artificial Intelligence Research.

# models folder copied and adapted from:
# https://github.com/kuangliu/pytorch-cifar/tree/master/models

from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement,
)

import argparse
import copy
import json
import os
import random
import shutil
import time
import warnings
import shlex
import sys
import os

from models import *

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', default='resnet50', type=str, metavar='ARCH', 
                        help='model architecture; select from '
                             'resnet18, resnet34, resnet50, resnet101, resnet152')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-scheduler', type=str, default='CosineAnnealingLR', 
                        help='learning rate scheduler (default: CosineAnnealingLR),'
                             ' either CosineAnnealingLR (which needs to set lr-tmax)'
                             ' or MultiStepLR (needs to set lr-schedule and lr-decay)')
    parser.add_argument('--lr-tmax', type=int, default=200, 
                        help='T_max (for CosineAnnealingLR only')
    parser.add_argument('--lr-schedule', nargs='+', type=int,
                        help='epochs for learning rate change,'
                             ' divide lr by lr-decay (for MultiStepLR only)')
    parser.add_argument('--lr-decay', default=0.2, type=float, 
                        help='learning rate decay rate at scheduled epochs'
                             ' (for MultiStepLR only)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)', 
                        dest='weight_decay')

    parser.add_argument('-m', '--dir-train-mask', default=None, type=str,
                        help='Boolean mask with True for indices to '
                             'train with and false for indices to skip.')
    parser.add_argument('--train-labels', type=str, default=None,
                        help='DIR of training labels format: json filename2integer')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on the training (if mask is not None)'
                             ' or the validation set (if mask is None)')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')

    parser.add_argument('--prune-percent', default=0, type=int,
                        help='NN pruning percentage')
    parser.add_argument('--prune-schedule', nargs='+', type=int,
                        help='epoch numbers for NN pruning')
    parser.add_argument('--prune-rm-schedule', nargs='+', type=int,
                        help='epoch numbers for removing NN pruning')

    parser.add_argument('--make-train-label', action='store_true',
                        help='make new training labels by'
                             ' replacing noisy ones with model prediction')
    parser.add_argument('--label-outfile', type=str, default='',
                        help='file path to save the newly created training labels.'
                             ' format: json filename2integer')

    parser.add_argument('--dynamic-train-label', action='store_true',
                        help='replace noisy labels with model prediction'
                             ' for every epoch during a period')
    parser.add_argument('--dynamic-frac-start', default=0.5, type=float,
                        help='first epoch (in fraction) for dynamic training label')
    parser.add_argument('--dynamic-frac-end', default=0.9, type=float,
                        help='last epoch (in fraction) for dynamic training label')

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cv-seed', default=0, type=int,
                        help='seed for determining the cv folds. ')
    parser.add_argument('--cv', '--cv-fold', type=int, default=None,
                        metavar='N', help='The fold to holdout')
    parser.add_argument('--cvn', '--cv-n-folds', default=0, type=int,
                        metavar='N', help='The number of folds')

    parser.add_argument('--combine-folds', action='store_true', default=False,
                        help='Pass this flag and -a arch to combine probs from all'
                             'folds. You must pass -a and -cvn flags as well!')
    parser.add_argument('--combine-source-path', type=str, default='',
                        help='source folder path for combining folds')
    parser.add_argument('--combine-dest-path', type=str, default='',
                        help='output folder path for combining folds')

    parser.add_argument('--turn-off-save-checkpoint', action='store_true',
                        help='Prevents saving model at every epoch of training.')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--print-out-file', type=str, default='',
                        help='file path for print output')

    parser.add_argument('--save-loss', action='store_true',
                        help='flag to save sample losses during training')
    parser.add_argument('--save-proba', action='store_true',
                        help='flag to save model predictions during training')

    return parser



def main(args):
    original_stdout = sys.stdout
    if args.print_out_file:
        # Print to the specified file
        out_f = open(args.print_out_file, 'w')
        sys.stdout = out_f

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    start_global = time.time()
    num_classes = 100 if ('cifar100' in args.data.lower()) else 10
    args.num_classes = num_classes
    best_acc1 = 0
    use_crossval = args.cvn > 0
    use_mask = args.dir_train_mask is not None
    cv_fold = args.cv
    cv_n_folds = args.cvn
    class_weights = None

    if use_crossval and use_mask:
        raise ValueError(
            'Either args.cvn > 0 or dir-train-mask not None, but not both.')

    if args.dynamic_train_label and not use_mask:
        raise ValueError(
            'dir-train-mask cannot be None when dynamic-train-label is enabled.')

    # Lead model
    if args.arch=='resnet18':
        model = ResNet18(num_classes)
    elif args.arch=='resnet34':
        model = ResNet34(num_classes)
    elif args.arch=='resnet101':
        model = ResNet101(num_classes)
    elif args.arch=='resnet152':
        model = ResNet152(num_classes)
    else:
        model = ResNet50(num_classes)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.lr_scheduler=='CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.lr_tmax)
    else: # MultiStepLR
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_schedule, gamma=args.lr_decay)


    # resume from a checkpoint
    resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            resume = True
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # In case you load checkpoint from different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]),
    )

    if args.evaluate and use_mask:
        # When both true, evaluate the masked training dataset
        # Create the training set so that it does not have input randomness
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )


    # if training labels are provided use those instead of dataset labels
    if args.train_labels is not None:
        with open(args.train_labels, 'r') as rf:
            train_labels_dict = json.load(rf)
        train_dataset.imgs = [(fn, train_labels_dict[fn]) for fn, _ in train_dataset.imgs]
        train_dataset.samples = train_dataset.imgs

    dataset_name = 'cifar100' if ('cifar100' in args.data.lower()) else 'cifar10'
    if not use_mask:
        base_folder = ((os.path.split(args.train_labels)[-1])[:-5] if args.train_labels is not None 
                        else 'orig_labels').replace('.','_') + '/seed_'+str(args.seed)
        sp = '_prune' if (args.prune_percent>0) else ''
        checkpoint_folder = '{}_training{}/{}'.format(dataset_name, sp, base_folder)
    else:
        base_folder = os.path.join('', *(os.path.normpath(args.dir_train_mask).split(os.sep)[1:-1]))
        checkpoint_folder = '{}_training_masked/{}'.format(dataset_name, base_folder)

    # If training only on cross-validated portion & make val_set = train_holdout
    if use_crossval:
        checkpoint_fn = "model_{}__fold_{}__checkpoint.pth.tar".format(
            args.arch, cv_fold)
        print('Computing fold indices. This takes 15 seconds.')
        # Prepare labels
        labels = [label for img, label in datasets.ImageFolder(traindir).imgs]
        # Split train into train and holdout for particular cv_fold.
        kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                             random_state=args.cv_seed)
        cv_train_idx, cv_holdout_idx = (
            list(kf.split(range(len(labels)), labels))[cv_fold])
        # Separate datasets
        np.random.seed(args.cv_seed)
        holdout_dataset = copy.deepcopy(train_dataset)
        holdout_dataset.imgs = [train_dataset.imgs[i] for i in cv_holdout_idx]
        holdout_dataset.samples = holdout_dataset.imgs
        train_dataset.imgs = [train_dataset.imgs[i] for i in cv_train_idx]
        train_dataset.samples = train_dataset.imgs
        print('Train size:', len(cv_train_idx), len(train_dataset.imgs))
        print('Holdout size:', len(cv_holdout_idx), len(holdout_dataset.imgs))

    else:
        checkpoint_fn = "model_{}__checkpoint.pth.tar".format(args.arch)
        if use_mask and not args.make_train_label:
            checkpoint_fn = "model_{}__masked__checkpoint.pth.tar".format(
                args.arch)
            orig_class_counts = np.bincount(
                [lab for img, lab in datasets.ImageFolder(traindir).imgs],
                minlength=num_classes,
            )
            train_bool_mask = np.load(args.dir_train_mask)

            # Mask labels
            train_dataset.imgs = [img for i, img in
                                  enumerate(train_dataset.imgs) if train_bool_mask[i]]
            train_dataset.samples = train_dataset.imgs

            clean_class_counts = np.bincount(
                [lab for img, lab in train_dataset.imgs],
                minlength=num_classes,
            )
            # We divide by this, so make sure no count is zero
            clean_class_counts = np.asarray(
                [1 if z == 0 else z for z in clean_class_counts])
            print('Train size:', len(train_dataset.imgs))
            # Compute class weights to re-weight loss during training
            # Should use the confident joint to estimate the noise matrix then
            # class_weights = 1 / p(s=k, y=k) for each class k.
            # Here we approximate this with a simpler approach
            # class_weights = count(y=k) / count(s=k, y=k)
            class_weights = torch.Tensor(orig_class_counts / clean_class_counts)
            class_weights = class_weights.cuda(args.gpu)

            if args.dynamic_train_label:
                # Relabel marked-noisy labels to -1 (for dynamic training label use)
                train_dataset_dynamic = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
                    ]),
                )
                train_dataset_dynamic.imgs = [(fn, l if train_bool_mask[i] else -1) 
                    for i, (fn, l) in enumerate(train_dataset.imgs)]
                train_dataset_dynamic.samples = train_dataset_dynamic.imgs

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=True,  # Don't train on last batch: could be 1 noisy example.
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
    # define separate loss function for val set that does not use class_weights
    val_criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)

    if args.evaluate:
        if use_mask:
            # Calculate masked training accuracy
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            validate(train_loader, model, criterion, args)
        else:
            # Calculate test accuracy
            validate(val_loader, model, criterion, args)
        return

    if args.make_train_label:
        # Make new training labels by replacing marked-noisy ones with model prediction
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        make_train_label(train_loader, train_dataset, model, args)
        return

    if args.save_loss or args.save_proba:
        # Create dataloader (with no randomness) to track and record training process
        train_dataset_track = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]),
        )
        if args.train_labels is not None:
            with open(args.train_labels, 'r') as rf:
                train_labels_dict = json.load(rf)
            train_dataset_track.imgs = [(fn, train_labels_dict[fn]) for fn, _ in
                                        train_dataset_track.imgs]
            train_dataset_track.samples = train_dataset_track.imgs

        train_loader_track = torch.utils.data.DataLoader(
            train_dataset_track,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        # Create list for saving training process, loss or model prediction
        save_args = (dataset_name, base_folder, args)
        if args.save_loss:
            save_loss_criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
            sample_train_loss = [] if not resume else load_file_fn(True, 'loss', save_args)
        if args.save_proba:
            sample_train_proba = [] if not resume else load_file_fn(True, 'proba', save_args)

    train_dynamic = False 

    for epoch in range(args.start_epoch, args.epochs):

        # Check if need to turn dynamic label training on
        if args.dynamic_train_label and not train_dynamic:
            if ((epoch/args.epochs) >= args.dynamic_frac_start and 
                (epoch/args.epochs) <= args.dynamic_frac_end):
                print('start dynamic train label')
                train_dynamic = True
                # Create dataloader for dynamic training labels
                train_loader_dynamic = torch.utils.data.DataLoader(
                    train_dataset_dynamic,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=None,
                    drop_last=True,
                )
                class_weights = None
                criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)

        if train_dynamic:
            train(train_loader_dynamic, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)

        # Update learning rate
        lr_scheduler.step()

        if args.prune_percent > 0:
            # Apply iterative pruning
            iterative_pruning(model, epoch+1, args)

        if args.save_loss or args.save_proba:
            # Append the training process of the current epoch
            train_container = {
                'criterion': save_loss_criterion if args.save_loss else None,
                'sample_loss_container': sample_train_loss if args.save_loss else None,
                'sample_proba_container': sample_train_proba if args.save_proba else None,
            }
            save_process_fn(train_loader_track, model, args, train_container)

        # Evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # Remember best acc@1, model, and save checkpoint.
        is_best = acc1 > best_acc1
        best_acc1 = max(best_acc1, acc1)

        if not args.turn_off_save_checkpoint:
            model_to_save = model
            if args.prune_percent > 0:
                # Make a deep copy of the mode and remove its pruning 
                for _,module in model.named_modules():
                    if prune.is_pruned(module):
                        model_to_save = copy.deepcopy(model)
                        print('deepcopy unpruned model to save checkpoint')
                        remove_prune(model_to_save)
                        break

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_to_save.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                foldername=checkpoint_folder,
                filename=checkpoint_fn,
                cv_fold=cv_fold,
                use_mask=use_mask,
            )

    if use_crossval:
        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        if not args.turn_off_save_checkpoint:  # Load best of saved checkpoints
            print("=> loading best model_{}__fold_{}_best.pth.tar".format(
                args.arch, cv_fold))
            checkpoint = torch.load(
                checkpoint_folder + "/model_{}__fold_{}_best.pth.tar".format(
                    args.arch, cv_fold))
            model.load_state_dict(checkpoint['state_dict'])
        print("Running forward pass on holdout set of size:",
              len(holdout_dataset.imgs))
        probs = get_probs(holdout_loader, model, args)
        np.save(checkpoint_folder+'/model_{}__fold_{}__probs.npy'.format(
            args.arch, cv_fold),
                probs)
        np.save(checkpoint_folder+'/time_{}__fold_{}.npy'.format(
            args.arch, cv_fold), np.array([time.time()-start_global]))

    else:
        np.save(checkpoint_folder+'/time_{}.npy'.format(
            args.arch), np.array([time.time()-start_global]))

    # Remove pruning after training
    remove_prune(model)

    # Save the whole training process
    if args.save_loss:
        save_file_fn(np.array(sample_train_loss).T, True, 'loss', save_args)
    if args.save_proba:
        save_file_fn(np.array(sample_train_proba), True, 'proba', save_args)

    sys.stdout = original_stdout
    if args.print_out_file:
        # Close the specified print output file
        out_f.close()


# Record and append the current epoch's training process, loss or prediction
def save_process_fn(loader_track, model, args, containers):
    model.eval()
    if args.save_loss:
        losses = []
        criterion = containers['criterion']
        sample_loss_container = containers['sample_loss_container']
    if args.save_proba:
        outputs = []
        sample_proba_container = containers['sample_proba_container']

    with torch.no_grad():
        for i, (input, target) in enumerate(loader_track):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(input)

            if args.save_loss:
                losses.append(
                    criterion(output, target) if args.gpu is None else
                    criterion(output, target).cpu().numpy()
                )
            if args.save_proba:
                outputs.append(output)

    if args.save_loss:
        sample_loss_container.append(np.concatenate(tuple(losses)))
    if args.save_proba:
        sample_proba_container.append(np.concatenate([
            torch.nn.functional.softmax(z, dim=1) if args.gpu is None else
            torch.nn.functional.softmax(z, dim=1).cpu().numpy()
            for z in outputs
        ]))


# Get path of the training process file
def get_file_path(istrain, suffix, save_args):
    (dataset_name, base_folder, args) = save_args
    sp = '_prune' if (args.prune_percent>0) else ''
    sm = '_masked' if (args.dir_train_mask is not None) else ''
    folderpath = '{}_training{}{}_{}/{}'.format(dataset_name, sp, sm, suffix, base_folder)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    datatype = 'train' if istrain else 'test'
    sf = '__cvn_{}__fold_{}'.format(args.cvn, args.cv) if args.cv is not None else ''
    filename = 'model_{}_{}_{}{}.npy'.format(args.arch, datatype, suffix, sf)
    return os.path.join(folderpath, filename)

def save_file_fn(data_arr, *_args):
    np.save(get_file_path(*_args), data_arr)

def load_file_fn(*_args):
    file_path = get_file_path(*_args)
    if os.path.isfile(file_path):
        tmp = np.load(file_path)
        if isinstance(tmp, np.ndarray):
            return list(tmp)
    return []


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if batch is size 1, skip because batch-norm will fail
        if len(input) <= 1:
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        if -1 in target:
            mask = torch.ones_like(target)
            mask[target==-1] = 0
            pred = torch.argmax(output, dim=1)
            target = target*mask + pred*(1-mask)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))


def iterative_pruning(model, epoch, args):
    if epoch in args.prune_schedule:
        print('iterative_pruning, prune_layers')
        prune_layers(model, args.prune_percent)
    elif epoch in args.prune_rm_schedule:
        print('iterative_pruning, remove_prune')
        remove_prune(model)

def prune_layers(model, prune_percent):
    for _,module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percent/100.)

def remove_prune(model):
    for _,module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, 'weight')
            except: pass
            try: prune.remove(module, 'bias')
            except: pass


def get_probs(loader, model, args):
    # Switch to evaluate mode.
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / n_total), end="")
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs.append(model(input))

    # Prepare outputs as a single matrix
    probs = np.concatenate([
        torch.nn.functional.softmax(z, dim=1) if args.gpu is None else
        torch.nn.functional.softmax(z, dim=1).cpu().numpy()
        for z in outputs
    ])

    return probs


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    confu_mtx = torch.zeros(args.num_classes, args.num_classes)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            pred = torch.argmax(output, dim=1)
            for t,p in zip(target, pred):
                confu_mtx[t,p] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        class_acc = 100* confu_mtx.diagonal() / (confu_mtx.sum(axis=1) + 1e-15)
        avgAcc = class_acc.mean()
        minAcc = class_acc.min()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} AccAvg {avgAcc:.3f} AccMin {minAcc:.3f}'
              .format(top1=top1, top5=top5, avgAcc=avgAcc, minAcc=minAcc))

    return top1.avg


# Make new label file that replaces marked-noisy labels (False in mask) with model's prediction
def make_train_label(data_loader, dataset, model, args):
    model.eval()
    pred_all = []
    target_all = []

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            target_all.append(
                target if args.gpu is None else 
                target.cpu().numpy()
            )

            output = model(input)
            pred = torch.argmax(output, dim=1)
            pred_all.append(
                pred if args.gpu is None else 
                pred.cpu().numpy()
            )

    pred_all = np.concatenate(tuple(pred_all))
    target_all = np.concatenate(tuple(target_all))
    train_bool_mask = np.load(args.dir_train_mask)

    new_labels = dict(zip([fn for fn, _ in dataset.imgs], 
        [int(target_all[i] if train_bool_mask[i] else pred_all[i]) for i in range(len(pred_all))]))

    with open(args.label_outfile, 'w') as wf:
        wf.write(json.dumps(new_labels))
    return


def save_checkpoint(state, is_best, foldername='.', filename='checkpoint.pth.tar', cv_fold=None,
                    use_mask=False):
    torch.save(state, foldername+'/'+filename)
    if is_best:
        sm = "__masked" if use_mask else ""
        sf = "__fold_{}".format(cv_fold) if cv_fold is not None else ""
        wfn = 'model_{}{}{}_best.pth.tar'.format(state['arch'], sm, sf)
        shutil.copyfile(foldername+'/'+filename, foldername+'/'+wfn)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
    specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def combine_folds(args):
    dataset_name = 'cifar100' if ('cifar100' in args.combine_source_path.lower()) else 'cifar10'
    num_classes = 100 if dataset_name=='cifar100' else 10
    wfn = dataset_name + '__train__model_{}__pyx.npy'.format(args.arch)
    print('Make sure you specified the model architecture with flag -a.')
    print('This method will overwrite file: {}'.format(wfn))
    print('Computing fold indices. This takes 15 seconds.')
    # Prepare labels
    labels = [label for img, label in
              datasets.ImageFolder(os.path.join(args.data, "train/")).imgs]
    # Initialize pyx array (output of trained network)
    pyx = np.empty((len(labels), num_classes))

    # Split train into train and holdout for each cv_fold.
    kf = StratifiedKFold(n_splits=args.cvn, shuffle=True,
                         random_state=args.cv_seed)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(
            kf.split(range(len(labels)), labels)):
        probs = np.load(os.path.join(args.combine_source_path, 'model_{}__fold_{}__probs.npy'.format(args.arch, k)))
        pyx[cv_holdout_idx] = probs[:, :num_classes]
    print('Writing final predicted probabilities.')
    np.save(os.path.join(args.combine_dest_path, wfn), pyx)

    # Compute overall accuracy
    print('Computing Accuracy.', flush=True)
    acc = sum(np.array(labels) == np.argmax(pyx, axis=1)) / float(len(labels))
    print('Accuracy: {:.25}'.format(acc))



if __name__ == '__main__':
    # if use Slurm array, can generate a dictionary of tasks using generate_commands.ipynb
    # with array ID as the key and a list of command strings as the value. 
    # The dictionary name is task_assignment by default.
    # Copy task_assignment to tasks.py
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        from tasks import task_assignment
        for task_str in task_assignment[task_id]:
            if '>' in task_str:
                arg_str, print_out_file = task_str.split('>')
            else:
                arg_str = task_str
            parser = get_parser()
            arg_parser = parser.parse_args((shlex.split(arg_str)))
            if '>' in task_str:
                arg_parser.print_out_file = ''.join(print_out_file.split(' '))

            if arg_parser.combine_folds:
                combine_folds(arg_parser)
            else:
                main(arg_parser)

    # python cifar10_train.py --arch resnet50 ... DATAPATH > logfile
    # Example commands can be found in generate_commands.ipynb
    # Need to add "python cifar10_train.py " in front of the generated commands.
    else:
        parser = get_parser()
        arg_parser = parser.parse_args()
        if arg_parser.combine_folds:
            combine_folds(arg_parser)
        else:
            main(arg_parser)


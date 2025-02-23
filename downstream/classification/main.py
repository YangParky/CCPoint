#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Xiaoyang Xiao
@Contact: lopeture@stu.xjtu.edu.cn
@File: main.py
@Time: 2024.4.29
"""

from __future__ import print_function
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import ModelNet40, ModelNet40Subset, ScanObjectNN
from model import PointNet_cls, DGCNN_cls
from util import get_time, cal_loss, IOStream
from parser import args


def _init_():
    if not os.path.exists('./output'):
        os.makedirs('./output')
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp)
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'models'):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'models')

    os.system('cp ./downstream/classification/data.py ./output/finetune/' + args.dataset + '/' + args.model + '/'  + args.exp + '/' + 'data.py')
    os.system('cp ./downstream/classification/main.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'main.py')
    os.system('cp ./downstream/classification/model.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'model.py')
    os.system('cp ./downstream/classification/parser.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'parser.py')
    os.system('cp ./downstream/classification/util.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'util.py')


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    dist_url = f'tcp://127.0.0.1:{args.master_port}'

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(backend=args.backend, init_method=dist_url, rank=rank, world_size=args.world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank):
    setup(rank)

    io = IOStream('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/run.log', rank=0)
    if args.dataset == 'modelnet40':
        if args.percent_train:
            train_set = ModelNet40Subset(partition='train',
                                         num_points=args.num_points,
                                         normalize=args.normalize_input,
                                         percent=args.percent_train)
            train_sampler = DistributedSampler(train_set,
                                               num_replicas=args.world_size,
                                               rank=rank)
            test_set = ModelNet40Subset(partition='test',
                                        num_points=args.num_points,
                                        normalize=args.normalize_input)
            test_sampler = DistributedSampler(test_set,
                                              num_replicas=args.world_size,
                                              rank=rank)

            train_loader = DataLoader(train_set,
                                      sampler=train_sampler,
                                      num_workers=4,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      drop_last=False)
            test_loader = DataLoader(test_set,
                                     sampler=test_sampler,
                                     num_workers=4,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     drop_last=False)
        else:
            train_set = ModelNet40(partition='train',
                                   num_points=args.num_points,
                                   normalize=args.normalize_input)
            train_sampler = DistributedSampler(train_set,
                                               num_replicas=args.world_size,
                                               rank=rank)
            test_set = ModelNet40(partition='test',
                                  num_points=args.num_points,
                                  normalize=args.normalize_input)
            test_sampler = DistributedSampler(test_set,
                                             num_replicas=args.world_size,
                                             rank=rank)

            train_loader = DataLoader(train_set,
                                      sampler=train_sampler,
                                      num_workers=4,
                                      batch_size=args.batch_size // args.world_size,
                                      shuffle=False,
                                      drop_last=False)
            test_loader = DataLoader(test_set,
                                     sampler=test_sampler,
                                     num_workers=4,
                                     batch_size=args.test_batch_size // args.world_size,
                                     shuffle=False,
                                     drop_last=False)
    else:

        train_set = ScanObjectNN(partition='training',
                                 num_points=args.num_points)
        train_sampler = DistributedSampler(train_set,
                                           num_replicas=args.world_size,
                                           rank=rank)
        test_set = ScanObjectNN(partition='test',
                                num_points=args.num_points)
        test_sampler = DistributedSampler(test_set,
                                          num_replicas=args.world_size,
                                          rank=rank)

        train_loader = DataLoader(train_set,
                                  sampler=train_sampler,
                                  num_workers=4,
                                  batch_size=args.batch_size // args.world_size,
                                  shuffle=False,
                                  drop_last=False)
        test_loader = DataLoader(test_set,
                                 sampler=test_sampler,
                                 num_workers=4,
                                 batch_size=args.test_batch_size // args.world_size,
                                 shuffle=False,
                                 drop_last=False)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    #Try to load models
    if args.model == 'pointnet_cls':
        model = PointNet_cls(args, output_channels=args.num_classes).to(rank)
    elif args.model == 'dgcnn_cls':
        model = DGCNN_cls(args, output_channels=args.num_classes).to(rank)
    else:
        raise Exception('Not implemented')
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    io.cprint(str(model))

    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.pretrain_path:
        if os.path.isfile(args.pretrain_path):
            pretrained_dict = torch.load(args.pretrain_path, map_location='cpu')
            pretrained_dict = pretrained_dict['point_model']
            pretrained_dict = {key.replace('module.encoder.', 'module.'): value for key, value in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=False)

            io.cprint(get_time() + "Loaded checkpoint '{}'".format(args.pretrain_path))
            del pretrained_dict

    if args.use_sgd:
        io.cprint(get_time() + 'Use SGD and CosineAnnealingLR ')
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr)
    else:
        io.cprint(get_time() + 'Use Adam and CosineAnnealingLR ')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr/100)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epoch):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(rank), label.to(rank).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, Batch %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' \
                 % (epoch, len(train_loader), train_loss*1.0/count, metrics.accuracy_score(train_true, train_pred),
                    metrics.balanced_accuracy_score(train_true, train_pred))
        io.cprint(get_time() + outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(rank), label.to(rank).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best test acc: %.6f' % \
                 (epoch, test_loss*1.0/count, test_acc, avg_per_class_acc, best_test_acc)
        io.cprint(get_time() + outstr)

        if rank == 0:
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), './output/finetune/%s/%s/%s/models/model.t7' % (args.dataset, args.model,  args.exp))

                io.cprint(get_time() + '==> Saving the best Model...')

    io.close()
    cleanup()


def test():

    test_loader = DataLoader(ModelNet40(partition='test',
                                        num_points=args.num_points),
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             drop_last=False)

    #Try to load models
    model = DGCNN_cls(args).to(args.cuda)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(args.cuda), label.to(args.cuda).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(get_time() + outstr)

    io.close()
    cleanup()


if __name__ == "__main__":

    _init_()

    io = IOStream('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/run.log', rank=0)
    io.cprint(get_time() + str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available() and torch.cuda.device_count() >= 1
    torch.manual_seed(args.seed)

    if not args.eval:
        if args.cuda:
            io.cprint(get_time() + 'CUDA is available! Using %d GPUs for DDP training' % args.world_size)
            io.close()

            torch.cuda.manual_seed(args.seed)
            mp.spawn(train, nprocs=args.world_size)
        else:
            io.cprint('Using CPU')
    else:
        test()

from __future__ import print_function
import os
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import ShapeNetPart, PartNormalDataset
from model import DGCNN_partseg
from parser import args
from util import cal_loss, get_time, IOStream


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def _init_():
    if not os.path.exists('./output/finetune/'):
        os.makedirs('./output/finetune/')
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp)
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/models'):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/models')
    os.system('cp ./downstream/segmentation/data.py ./output/finetune/' + args.dataset + '/' + args.model + '/'  + args.exp + '/' + 'data.py')
    os.system('cp ./downstream/segmentation/main_partseg.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'main_partseg.py')
    os.system('cp ./downstream/segmentation/model.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'model.py')
    os.system('cp ./downstream/segmentation/parser.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'parser.py')
    os.system('cp ./downstream/segmentation/util.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'util.py')


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    dist_url = f'tcp://127.0.0.1:{args.master_port}'

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(backend=args.backend, init_method=dist_url, rank=rank, world_size=args.world_size)


def cleanup():
    dist.destroy_process_group()


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    cat_ious = [[] for i in range(16)]
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        cat_ious[label[shape_idx]].append(np.mean(part_ious))
    # for item in cat_ious:
    #     print(np.mean(item), end=" ")

    return shape_ious


def train(rank):
    setup(rank)

    io = IOStream('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/run.log', rank=rank)

    train_set = ShapeNetPart(partition='trainval',
                             num_points=args.num_points,
                             class_choice=args.class_choice)
    test_set = ShapeNetPart(partition='test',
                            num_points=args.num_points,
                            class_choice=args.class_choice)

    train_sampler = DistributedSampler(train_set,
                                       num_replicas=args.world_size,
                                       rank=rank)
    test_sampler = DistributedSampler(test_set,
                                      num_replicas=args.world_size,
                                      rank=rank)
    samples_per_gpu = args.batch_size // args.world_size

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=samples_per_gpu,
                              shuffle=False,
                              # num_workers=4,
                              drop_last=False)
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             batch_size=samples_per_gpu,
                             shuffle=False,
                             # num_workers=4,
                             drop_last=False)
    args.rank = rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    if args.model == 'dgcnn_part':
        model = DGCNN_partseg(args, seg_num_all).to(rank)
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
        io.cprint(get_time() + 'Use SGD')
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        io.cprint(get_time() + 'Use Adam')
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'Cos':
        io.cprint(get_time() + 'Use Cos')
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    elif args.scheduler == 'Step':
        io.cprint(get_time() + 'Use Step')
        scheduler = StepLR(opt, step_size=40, gamma=0.5)

    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epoch):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in train_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(rank), label_one_hot.to(rank), seg.to(rank)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.mean().backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))   # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))  # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'Cos':
            scheduler.step()
        elif args.scheduler == 'Step':
            if opt.param_groups[0]['lr'] > 0.9e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 0.9e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 0.9e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        outstr = 'Train %d, Batch %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % \
                 (epoch, len(train_loader), train_loss*1.0/count, train_acc, avg_per_class_acc, np.mean(train_ious))
        io.cprint(get_time() + outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(rank), label_one_hot.to(rank), seg.to(rank)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, best iou: %.6f' % \
                 (epoch, test_loss*1.0/count, test_acc, avg_per_class_acc, np.mean(test_ious), best_test_iou)

        io.cprint(get_time() + outstr)

        if rank == 0:
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                torch.save(model.state_dict(), './output/finetune/%s/%s/%s/models/model.t7' % (args.dataset, args.model, args.exp))


def test():
    test_loader = DataLoader(ShapeNetPart(partition='test',
                                          num_points=args.num_points,
                                          class_choice=args.class_choice,
                                          class_test=args.class_test),
                             batch_size=args.test_batch_size,
                             shuffle=True,
                             drop_last=False)

    device = args.gpu_id

    #Try to load models
    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index
    if args.model == 'dgcnn':
        model = DGCNN_partseg(args, seg_num_all).to(device)
    else:
        raise Exception('Not implemented')

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []

    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % \
             (test_acc, avg_per_class_acc, np.mean(test_ious))
    io.cprint(get_time() + outstr)


if __name__ == '__main__':
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
            io.cprint(get_time() + 'Using CPU')
    else:
        # for c in ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
        #           'lamp', 'laptop', 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table']:
        #     args.class_test = c
        #     test()
        test()

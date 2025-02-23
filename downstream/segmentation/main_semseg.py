from __future__ import print_function
import os
import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics as metrics

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data import S3DIS
from model import DGCNN_semseg
from parser import args
from util import cal_loss, get_time, IOStream


def _init_():
    if not os.path.exists('./output/finetune/'):
        os.makedirs('./output/finetune/')
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp)
    if not os.path.exists('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/models'):
        os.makedirs('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/models')
    os.system('cp ./downstream/segmentation/data.py ./output/finetune/' + args.dataset + '/' + args.model + '/'  + args.exp + '/' + 'data.py')
    os.system('cp ./downstream/segmentation/main_semseg.py ./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/' + 'main_semseg.py')
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


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def train(rank):
    setup(rank)

    io = IOStream('./output/finetune/' + args.dataset + '/' + args.model + '/' + args.exp + '/run.log', rank=rank)

    train_set = S3DIS(partition='train',
                      num_points=args.num_points,
                      test_area=args.test_area)
    test_set = S3DIS(partition='test',
                     num_points=args.num_points,
                     test_area=args.test_area)

    train_sampler = DistributedSampler(train_set,
                                       num_replicas=args.world_size,
                                       rank=rank)
    test_sampler = DistributedSampler(test_set,
                                      num_replicas=args.world_size,
                                      rank=rank)

    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              num_workers=8,
                              batch_size=args.batch_size // args.world_size,
                              shuffle=False,
                              drop_last=False)
    test_loader = DataLoader(test_set,
                             sampler=test_sampler,
                             num_workers=8,
                             batch_size=args.test_batch_size // args.world_size,
                             shuffle=False,
                             drop_last=False)
    args.rank = rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    #Try to load models
    if args.model == 'dgcnn_sem':
        model = DGCNN_semseg(args).to(rank)
    else:
        raise Exception('Not implemented')
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    io.cprint(str(model))

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
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint(get_time() + 'Use Adam')
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'Cos':
        io.cprint(get_time() + 'Use Cos')
        scheduler = CosineAnnealingLR(opt, args.epoch, eta_min=args.lr)
    elif args.scheduler == 'Step':
        io.cprint(get_time() + 'Use Step')
        scheduler = StepLR(opt, step_size=20, gamma=0.5)

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

        for data, seg in train_loader:
            data, seg = data.to(rank), seg.to(rank)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss.mean().backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'Cos':
            scheduler.step()
        elif args.scheduler == 'Step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % \
                 (epoch, train_loss*1.0/count, train_acc, avg_per_class_acc, np.mean(train_ious))
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
        for data, seg in test_loader:
            data, seg = data.to(rank), seg.to(rank)
            data = data.permute(0, 2, 1)

            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, best iou: %.6f' % \
                 (epoch, test_loss*1.0/count, test_acc, avg_per_class_acc, np.mean(test_ious), best_test_iou)
        io.cprint(get_time() + outstr)

        if rank == 0:
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                torch.save(model.state_dict(), './output/finetune/%s/%s/%s/models/model_%s.t7' % (
                    args.dataset,
                    args.model,
                    args.exp,
                    args.test_area))


def test():

    device = args.gpu_id

    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            #Try to load models
            if args.model == 'dgcnn_semseg':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception('Not implemented')

            model.load_state_dict(torch.load(os.path.join(args.model_path, 'model_%s.t7' % test_area)))
            model = model.eval()

            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % \
                     (test_area, test_acc, avg_per_class_acc, np.mean(test_ious))
            io.cprint(get_time() + outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % \
                 (all_acc, avg_per_class_acc, np.mean(all_ious))
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

            torch.cuda.manual_seed(args.seed)
            mp.spawn(train, nprocs=args.world_size)
        else:
            io.cprint(get_time() + 'Using CPU')
    else:
        test()

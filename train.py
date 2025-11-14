from __future__ import print_function
import os
import torch
import numpy as np
import time
import random

from sklearn.svm import SVC
from lightly import loss

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from extensions.chamfer_dist import ChamferDistanceL2

from dataset.data import ModelNet40SVM, ScanObjectNNSVM, ShapeNetRender_Corruption
from dataset import corrupt_utils as c_utils
from models.dgcnn import DGCNN_cls, DGCNN_partseg, DGCNN_semseg
from models.pointnet import PointNet_cls, PointNet_partseg
from models.ccpoint import CCPoint
from util import IOStream, AverageMeter
from parser import args


def _init_():
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output/pretrain/'  + args.model + '/' + args.exp):
        os.makedirs('output/pretrain/'  + args.model + '/' + args.exp)
    if not os.path.exists('output/pretrain/'  + args.model + '/' + args.exp + '/' + 'models'):
        os.makedirs('output/pretrain/'  + args.model + '/' + args.exp + '/' + 'models')

    os.system('cp ./dataset/data.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'data.py')
    os.system('cp ./dataset/corrupt_utils.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'corrupt_utils.py')
    os.system('cp ./dataset/data_utils.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'data_utils.py')
    os.system('cp ./models/dgcnn.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'dgcnn.py')
    os.system('cp ./models/ccpoint.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'ccpoint.py')
    os.system('cp ./models/foldingnet.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'foldingnet.py')
    os.system('cp ./models/mlp.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'mlp.py')
    os.system('cp parser.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'parser.py')
    os.system('cp train.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'train.py')
    os.system('cp util.py output/pretrain/'  + args.model + '/' + args.exp + '/' + 'util.py')


def scale_lr_lambda(init_lr, batch_size, base_batch_size=32, model_type='dgcnn'):
    """function for scaling LR according to batch size."""
    if 'dgcnn' in model_type:
        return init_lr * (batch_size / base_batch_size)
    else:
        return init_lr * (batch_size / base_batch_size) * 0.1


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    dist_url = f"tcp://127.0.0.1:{args.master_port}"

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(backend=args.backend, init_method=dist_url, rank=rank, world_size=args.world_size)


def cleanup():
    dist.destroy_process_group()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_time():
    time_now = time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(int(round(time.time()*1000))/1000))

    return time_now


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(rank):

    setup(rank)

    io = IOStream('output/pretrain/'  + args.model + '/' + args.exp + '/run.log', rank=rank)

    train_set = ShapeNetRender_Corruption(corrupt_type=args.corrupt_affine, affine_level=args.affine_level)
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)

    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=samples_per_gpu,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True
                              )

    # in DGCNN and DGCNN_partseg, args.rank is used to specify the device where get_graph_feature() are executed
    args.rank = rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # Try to load models
    if args.model == 'dgcnn_cls':
        point_model = CCPoint(args, backbone=DGCNN_cls(args)).to(rank)
    elif args.model == 'dgcnn_part':
        point_model = CCPoint(args, backbone=DGCNN_partseg(args)).to(rank)
    elif args.model == 'dgcnn_sem':
        point_model = CCPoint(args, backbone=DGCNN_semseg(args)).to(rank)
    elif args.model == 'pointnet_cls':
        point_model = CCPoint(args, backbone=PointNet_cls(args)).to(rank)
    elif args.model == 'pointnet_part':
        point_model = CCPoint(args, backbone=PointNet_partseg(args)).to(rank)
    else:
        raise Exception("Not implemented")
    point_model = DDP(point_model, device_ids=[rank], find_unused_parameters=False)

    if args.world_size > 1:
        point_model = nn.SyncBatchNorm.convert_sync_batchnorm(point_model)
    io.cprint(str(point_model))

    corrupt_functions = {
        'add_global': c_utils.corrupt_add_global,
        'add_local': c_utils.corrupt_add_local,
        'dropout_global': c_utils.corrupt_dropout_global,
        'dropout_local': c_utils.corrupt_dropout_local,
        'cutout': c_utils.corrupt_cutout,
        'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
    }

    # NOTE: combine parameters for different models
    para = list(point_model.parameters())

    total_params = count_parameters(point_model)
    print(f"Total Parameters: {total_params}")

    lr = scale_lr_lambda(args.lr, args.batch_size, base_batch_size=64, model_type=args.model)

    if args.use_sgd:
        io.cprint(get_time() + "Use SGD and CosineAnnealingLR ")
        optimizer = optim.SGD(point_model.parameters(), lr=lr*100, momentum=args.momentum, weight_decay=1e-6)
    else:
        io.cprint(get_time() + "Use Adam and CosineAnnealingLR ")
        optimizer = optim.Adam(para, lr=lr,  weight_decay=1e-6)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            point_model.load_state_dict(checkpoint['point_model'])
            best_acc = checkpoint['best_acc']

            io.cprint(get_time() + "Loaded checkpoint '{}' (epoch {})".
                      format(args.resume, checkpoint['epoch']))

            del checkpoint
    else:
        best_acc = 0

    if args.scheduler == 'Cos':
        io.cprint(get_time() + 'Use Cos')
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    elif args.scheduler == 'Step':
        io.cprint(get_time() + 'Use Step')
        lr_scheduler = StepLR(optimizer, step_size=args.step, gamma=0.5)

    criterion1 = loss.NegativeCosineSimilarity().to(rank)
    criterion2 = ChamferDistanceL2().to(rank)
    alpha = args.alpha
    beta = args.beta

    if args.corrupt_extra == 'None':
        corrupt_extra = 'None'
    elif args.corrupt_extra == 'random_noise':
        corrupt_extra = random.choices(['add_global', 'add_local'])[0]
    elif args.corrupt_extra == 'random_masking':
        corrupt_extra = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
    elif 'dropout' in args.corrupt_extra or 'add' in args.corrupt_extra or 'jitter' in args.corrupt_extra:
        corrupt_extra = args.corrupt_extra
    elif args.corrupt_extra == 'random_mixing':
        pass
    else:
        raise Exception('Not implemented')

    if args.extra_level == -1:
        levels = [0, 1, 2, 3, 4]
        extra_level = random.choice(levels)
    elif args.extra_level == -2:
        levels = [1, 2, 4]
        extra_level = random.choice(levels)
    else:
        extra_level = args.extra_level

    for epoch in range(args.start_epoch, args.epoch):
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        train_intra1_losses = AverageMeter()
        train_intra2_losses = AverageMeter()
        train_intra3_losses = AverageMeter()
        train_coarse_losses = AverageMeter()
        train_fine_losses = AverageMeter()

        # Require by DistributedSampler
        train_sampler.set_epoch(epoch)
        point_model.train()

        io.cprint(get_time() + f'Start training epoch: ({epoch}/{args.epoch})')
        for i, (point_t1, point_t2, point_cr) in enumerate(train_loader):
            # batch_start_time = time.time()
            batch_size = point_t1.size()[0]

            if args.corrupt_extra == 'random_mixing':
                corrupt_extra = random.choices(['add_global', 'add_local'])[0]
                point_cr = corrupt_functions[corrupt_extra](point_cr, extra_level)
                corrupt_extra = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
                point_cr = corrupt_functions[corrupt_extra](point_cr, extra_level)
            else:
                if corrupt_extra in corrupt_functions:
                    point_cr = corrupt_functions[corrupt_extra](point_cr, extra_level)
                else:
                    raise Exception('Not implemented')

            batch_start_time = time.time()

            point_t1 = point_t1.to(rank).transpose(2, 1).contiguous()
            point_t2 = point_t2.to(rank).transpose(2, 1).contiguous()
            point_cr = point_cr.to(rank).transpose(2, 1).contiguous()
            optimizer.zero_grad()

            point_proj1, point_prid1, point_proj2, point_prid2, corrup_proj, corrup_coarse, corrup_fine = \
                point_model(point_t1, point_t2, point_cr)

            loss_intra1 = (criterion1(point_prid1, point_proj2.detach()) +
                           criterion1(point_prid2, point_proj1.detach())) * 0.5
            loss_intra2 = (criterion1(point_prid1, corrup_proj.detach()) +
                           criterion1(point_prid2, corrup_proj.detach())) * 0.5
            loss_intra3 = (criterion1(corrup_proj, point_proj1.detach()) +
                           criterion1(corrup_proj, point_proj2.detach())) * 0.5
            loss_coarse = criterion2(corrup_coarse, point_t1.transpose(2, 1))
            loss_fine = criterion2(corrup_fine, point_t1.transpose(2, 1))

            total_loss = alpha * (loss_intra1 + loss_intra2 + loss_intra3) + beta * (loss_coarse + loss_fine)
            total_loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()

            train_losses.update(total_loss.item(), batch_size)
            train_intra1_losses.update(loss_intra1.item(), batch_size)
            train_intra2_losses.update(loss_intra2.item(), batch_size)
            train_intra3_losses.update(loss_intra3.item(), batch_size)
            train_coarse_losses.update(loss_coarse.item(), batch_size)
            train_fine_losses.update(loss_fine.item(), batch_size)

            # Record batch time and total frames
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            fps = batch_size / batch_time  # Frames per second

        # Log batch-level information
        outstr = 'Epoch (%d), Batch(%d/%d), loss: %.6f, intra1 loss: %.6f, intra2 loss: %.6f, ' \
                 'intra3 loss: %.6f, coarse loss: %.6f, fine_loss: %.6f, batch time: %.4f s, FPS: %.2f' % \
                 (epoch, i, len(train_loader), train_losses.avg, train_intra1_losses.avg, train_intra2_losses.avg,
                  train_intra3_losses.avg, train_coarse_losses.avg, train_fine_losses.avg, batch_time, fps)
        io.cprint(get_time() + outstr)

        # Testing_MN40
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024, cat=40), batch_size=32,
                                      shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024, cat=40), batch_size=32,
                                     shuffle=False)
        feats_train = []
        labels_train = []
        point_model.eval()
        for (data, label) in train_val_loader:
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model(data)[0]

            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels
        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []
        for data, label in test_val_loader:
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels
        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        model_tl = SVC(C=0.1, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test40_accuracy = model_tl.score(feats_test, labels_test)

        # Testing_MN10
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024, cat=10), batch_size=32,
                                      shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024, cat=10), batch_size=32,
                                     shuffle=False)

        feats_train = []
        labels_train = []
        point_model.eval()
        for (data, label) in train_val_loader:
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels
        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []
        for data, label in test_val_loader:
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels
        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        model_t2 = SVC(C=0.1, kernel='linear')
        model_t2.fit(feats_train, labels_train)
        test10_accuracy = model_t2.score(feats_test, labels_test)

        if epoch % args.eval_freq == 0:
            # Testing_Scan
            train_val_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=1024), batch_size=32, shuffle=True)
            test_val_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=1024), batch_size=32, shuffle=False)

            feats_train = []
            labels_train = []
            point_model.eval()
            for (data, label) in train_val_loader:
                labels = label.numpy().tolist()
                data = data.permute(0, 2, 1).to(rank)
                with torch.no_grad():
                    feats = point_model(data)[0]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_train.append(feat)
                labels_train += labels
            feats_train = np.array(feats_train)
            labels_train = np.array(labels_train)

            feats_test = []
            labels_test = []
            for data, label in test_val_loader:
                labels = label.numpy().tolist()
                data = data.permute(0, 2, 1).to(rank)
                with torch.no_grad():
                    feats = point_model(data)[0]
                feats = feats.detach().cpu().numpy()
                for feat in feats:
                    feats_test.append(feat)
                labels_test += labels
            feats_test = np.array(feats_test)
            labels_test = np.array(labels_test)
            model_t3 = SVC(C=0.1, kernel='linear')
            model_t3.fit(feats_train, labels_train)
            scan_accuracy = model_t3.score(feats_test, labels_test)

            io.cprint(get_time() +
                      f"Linear40 Accuracy : {test40_accuracy}, "
                      f"Linear10 Accuracy : {test10_accuracy}, "
                      f"LinearScan Accuracy : {scan_accuracy}, "
                      f"Best40 Accuracy : {best_acc}")
        else:
            io.cprint(get_time() +
                      f"Linear40 Accuracy : {test40_accuracy}, "
                      f"Best40 Accuracy : {best_acc}")

        if rank == 0:
            if test40_accuracy > best_acc:
                best_acc = test40_accuracy

                model_file = os.path.join(f'output/pretrain/{args.model}/{args.exp}/models/checkpoint_best.pth.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'point_model': point_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, filename=model_file)
                io.cprint(get_time() + '==> Saving Best Model...')

            if (epoch+1) % 10 == 0:
                model_file = os.path.join(f'output/pretrain/{args.model}/{args.exp}/models/checkpoint_{epoch+1}.pth.tar')
            else:
                model_file = os.path.join(f'output/pretrain/{args.model}/{args.exp}/models/checkpoint_last.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'point_model': point_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename=model_file)


            io.cprint(get_time() + '==> Saving Last Model...')

    io.close()
    cleanup()


if __name__ == "__main__":
    _init_()

    io = IOStream('output/pretrain/' + '/' + args.model + '/' + args.exp + '/run.log', rank=0)
    io.cprint(get_time() + str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available() and torch.cuda.device_count() >= 1
    args.cuda = True

    torch.manual_seed(args.seed)

    if args.cuda:
        io.cprint(get_time() + 'CUDA is available! Using %d GPUs for DDP training' % args.world_size)
        io.close()

        torch.cuda.manual_seed(args.seed)
        mp.spawn(train, nprocs=args.world_size)
    else:
        io.cprint(get_time() + 'CUDA is unavailable! Exit')
        io.close()
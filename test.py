from __future__ import print_function
import torch
import numpy as np
import torch.multiprocessing as mp

from sklearn.svm import SVC
from torch.utils.data import DataLoader

from dataset.data import ModelNet40SVM, ScanObjectNNSVM
from models.dgcnn import DGCNN_cls, DGCNN_partseg, DGCNN_semseg
from models.pointnet import PointNet_cls
from models.ccpoint import CCPoint
from util import IOStream, get_time
from parser import args

torch.autograd.set_detect_anomaly(True)


def test(rank):

    io = IOStream('output/pretrain/' + args.model + '/' + args.exp + '/run.log', rank=0)

    # Try to load models
    if args.model == 'dgcnn_cls':
        point_model = CCPoint(args, backbone=DGCNN_cls(args)).to(rank)
    elif args.model == 'dgcnn_part':
        point_model = CCPoint(args, backbone=DGCNN_partseg(args)).to(rank)
    elif args.model == 'dgcnn_sem':
        point_model = CCPoint(args, backbone=DGCNN_semseg(args)).to(rank)
    elif args.model == 'pointnet_cls':
        point_model = CCPoint(args, backbone=PointNet_cls(args)).to(rank)
    else:
        raise Exception('Not implemented')

    # model_path = './output/pointnet_cls/dgcnn_cls/models/checkpoint_best.pth.tar'
    model_path = './output/pretrain/dgcnn_cls/models/checkpoint_best.pth.tar'

    map_location = torch.device('cuda:%d' % rank)

    ckpt = torch.load(model_path, map_location=map_location)['point_model']
    ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
    point_model.load_state_dict(ckpt, strict=True)

    io.cprint(get_time() + 'Model Loaded !!')

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
              f'Linear40 Accuracy : {test40_accuracy}, '
              f'Linear10 Accuracy : {test10_accuracy}, '
              f'LinearScan Accuracy : {scan_accuracy}, '
              )

    io.close()


if __name__ == '__main__':

    io = IOStream('output/pretrain/' + args.model + '/' + args.exp + '/run.log', rank=0)
    io.cprint(get_time() + str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available() and torch.cuda.device_count() >= 1
    torch.manual_seed(args.seed)

    if args.cuda:
        io.cprint(get_time() + 'CUDA is available! Using %d GPUs for DDP training' % args.world_size)
        io.close()

        torch.cuda.manual_seed(args.seed)
        mp.spawn(test, nprocs=args.world_size)
    else:
        io.cprint(get_time() + 'CUDA is unavailable! Exit')
        io.close()
import os
import sys
import argparse
import random
import torch
import pandas as pd

from torch.utils.data import DataLoader
from pyntcloud import PyntCloud

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from models.ccpoint import CCPoint
from models.dgcnn import DGCNN_cls
from models.pointnet import PointNet_cls
from models.foldingnet import FoldingNet
from dataset.data import ShapeNetRender_Corruption
from dataset import corrupt_utils as c_utils

# Training settings
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp', type=str, default='tab9', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn_cls', metavar='N',
                    choices=['dgcnn_cls', 'dgcnn_part', 'dgcnn_sem', 'pointnet_cls', 'pointnet_part', 'moco_dgcnn_cls'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epoch', type=int, default=100, metavar='N',
                    help='Number of episode to train ')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='Number of episode to train ')
parser.add_argument('--use_sgd', action='store_true',
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate (default: 0.001 for dgcnn, 0.0001 for pointnet)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='Cos', metavar='N',
                    choices=['Cos', 'Step'],
                    help='Scheduler to use, [Cos, Step]')
parser.add_argument('--step', type=int, default=40,
                    help='lr decay step')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='evaluate the model')
parser.add_argument('--num_points', type=int, default=2048,
                    help='Num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--disable_amp', action='store_true',
                    help='Disable mixed-precision training (requires more memory and compute)')
parser.add_argument('--eval_freq', default=5, type=int,
                    help='Eval modelnet10 and scanobjectnn frequency')
parser.add_argument('--resume', default='', type=str,
                    help='Path to latest checkpoint (default: none)')

# Corruption level
parser.add_argument('--corrupt_affine', type=str, default='affine_r5',
                    help='Affine corruption including affine_r5, affine_r3')
parser.add_argument('--affine_level', type=int, default=-1,
                    help='The level of affine 0-4, -1 stands for random sampling, -2 stands for the given sampling')
parser.add_argument('--corrupt_extra', type=str, default='None',
                    help='Extra corruption including random_noise, random_masking')
parser.add_argument('--extra_level', type=int, default=-1,
                    help='The level of extra corruption 0-4, -1 stand for random sampling, -2 stands for the given sampling')
parser.add_argument('--neg_lambda', type=float, default=0.2,
                    help='The value of negative lambda')

# Training on single GPU device
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='Enables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='Specify the GPU device to train of finetune model')

# Distributed training on multiple GPUs
parser.add_argument('--rank', type=int, default=-1,
                    help='The rank for current GPU or process, ususally one process per GPU')
parser.add_argument('--backend', type=str, default='nccl',
                    help='DDP communication backend')
parser.add_argument('--world_size', type=int, default=1,
                    help='Number of GPUs')
parser.add_argument('--master_addr', type=str, default='localhost',
                    help='Ip of master node')
parser.add_argument('--master_port', type=str, default='12355',
                    help='Port of master node')
args = parser.parse_args()

device = torch.device('cuda')
save_path = './output/visualization/reconstruction/'
resume_path = './output/pretrain/dgcnn_cls/models/checkpoint_last.pth.tar'

args.corrupt_affine = 'None'

test_set = ShapeNetRender_Corruption(corrupt_type=args.corrupt_affine,
                                     affine_level=args.affine_level)
test_loader = DataLoader(test_set,
                         batch_size=2,
                         shuffle=True,
                         num_workers=8,
                         pin_memory=True
                         )

model = CCPoint(args, backbone=DGCNN_cls(args)).to(device)

checkpoint = torch.load(resume_path, map_location='cpu')
checkpoint = {key.replace('module.', ''): value for key, value in checkpoint['point_model'].items()}
model.load_state_dict(checkpoint, strict=True)
print("Loaded checkpoint '{}' successfully.".format(resume_path))
del checkpoint

corrupt_functions = {
    'add_global': c_utils.corrupt_add_global,
    'add_local': c_utils.corrupt_add_local,
    'dropout_global': c_utils.corrupt_dropout_global,
    'dropout_local': c_utils.corrupt_dropout_local,
    'cutout': c_utils.corrupt_cutout,
    'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
}
# corrupt_extra = 'random_mixing'
corrupt_extra = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
extra_level = random.choice([1, 2, 4])
model.eval()

with torch.no_grad():
    for i, (point_t1, point_t2, point_cr) in enumerate(test_loader):

        if corrupt_extra == 'random_mixing':
            corrupt_extra = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
            point_cr = corrupt_functions[corrupt_extra](point_cr, extra_level)
            corrupt_extra = random.choices(['add_global', 'add_local'])[0]
            point_cr = corrupt_functions[corrupt_extra](point_cr.cpu(), extra_level)
        else:
            if corrupt_extra in corrupt_functions:
                point_cr = corrupt_functions[corrupt_extra](point_cr, extra_level)
            else:
                raise Exception('Not implemented')

        point_t1 = point_t1.to(device).transpose(2, 1).contiguous()
        point_t2 = point_t2.to(device).transpose(2, 1).contiguous()
        point_cr = point_cr.to(device).transpose(2, 1).contiguous()

        # point_proj1, point_prid1, point_proj2, point_prid2, corrup_proj, corrup_coarse, corrup_fine = \
        #     model(point_t1, point_t2, point_cr)

        corrup_coarse, corrup_fine = \
            model(point_t1, point_t2, point_t1)

        point_t1 = point_t1[0].permute(1,0).cpu().numpy()
        point_cr = point_cr[0].permute(1,0).cpu().numpy()
        corrup_coarse = corrup_coarse[0].cpu().numpy()
        corrup_fine = corrup_fine[0].cpu().numpy()

        print(point_t1.shape, corrup_coarse.shape, corrup_fine.shape)

        d = {'x': point_t1[:, 0], 'y': point_t1[:, 1], 'z': point_t1[:, 2]}
        cloud = PyntCloud(pd.DataFrame(data=d))
        cloud.to_file(save_path + str(i) + '_raw_point.ply')

        d = {'x': point_cr[:, 0], 'y': point_cr[:, 1], 'z': point_cr[:, 2]}
        cloud = PyntCloud(pd.DataFrame(data=d))
        cloud.to_file(save_path + str(i) + '_cor_point.ply')

        d = {'x': corrup_coarse[:, 0], 'y': corrup_coarse[:, 1], 'z': corrup_coarse[:, 2]}
        cloud = PyntCloud(pd.DataFrame(data=d))
        cloud.to_file(save_path + str(i) + '_rec_coarse.ply')

        d = {'x': corrup_fine[:, 0], 'y': corrup_fine[:, 1], 'z': corrup_fine[:, 2]}
        cloud = PyntCloud(pd.DataFrame(data=d))
        cloud.to_file(save_path + str(i) + '_rec_fine.ply')

        if i > 20:
            break
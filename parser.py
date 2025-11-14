import argparse

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
parser.add_argument('--alpha', type=float, default=1,
                    help='The loss weight')
parser.add_argument('--beta', type=float, default=1,
                    help='The loss weight')
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

import argparse

# Training settings
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--exp', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn_part', metavar='N',
                    choices=['dgcnn_part', 'dgcnn_sem'],
                    help='Model to use, [dgcnn_part]')
parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                    choices=['shapenetpart', 's3dis'])

# For part segmentation
parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                             'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                             'motor', 'mug', 'pistol', 'rocket', 'skateboard',
                             'table'])
parser.add_argument('--class_test', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                             'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                             'motor', 'mug', 'pistol', 'rocket', 'skateboard',
                             'table'])
# For semantic segmentation
parser.add_argument('--test_area', type=str, default='5', metavar='N',
                    choices=['1', '2', '3', '4', '5', '6', 'all'])

parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epoch', type=int, default=300, metavar='N', choices=[300, 200, 100],
                    help='Number of episode to train ')
parser.add_argument('--use_sgd', action='store_true',
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='Cos', metavar='N',
                    choices=['Cos', 'Step'],
                    help='Scheduler to use, [Cos, Step]')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluate the model')
parser.add_argument('--num_points', type=int, default=2048, choices=[1024, 2048, 4096],
                    help='Num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=40, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--disable_amp', action='store_true',
                    help='Disable mixed-precision training (requires more memory and compute)')
parser.add_argument('--eval_freq', default=5, type=int,
                    help='Eval modelnet10 and scanobjectnn frequency')
parser.add_argument('--resume', default='', type=str,
                    help='Path to latest checkpoint (default: none)')

# Finetune
parser.add_argument('--pretrain_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Model path')
parser.add_argument('--normalize_input', type=int, default=0,
                    help='Whether to normalize input dataset to a unit space')
parser.add_argument('--percent_train', type=float, default=None,
                    help='Only use part of training dataset')

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

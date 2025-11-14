import os
import sys
import argparse
import torch

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from models.ccpoint import CCPoint
from models.dgcnn import DGCNN_cls
from dataset.data import ModelNet40SVM

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

save_path = './output/visualization/tsne/'

resume_path010 = './output/pretrain/dgcnn_cls/tab11/models/checkpoint_10.pth.tar'
resume_path050 = './output/pretrain/dgcnn_cls/tab11/models/checkpoint_50.pth.tar'
resume_path100 = './output/pretrain/dgcnn_cls/tab11/models/checkpoint_best.pth.tar'

test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024, cat=10),
                             batch_size=32,
                             shuffle=False)
print("Loaded ModelNet10 test dataset.")

# model_000 = CCPoint(args, backbone=DGCNN_cls(args)).to(device)
# print("Init model successfully.")

model_010 = CCPoint(args, backbone=DGCNN_cls(args)).to(device)
# checkpoint = torch.load(resume_path010, map_location='cpu')['point_model']
# checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
# model_010.load_state_dict(checkpoint, strict=True)
# del checkpoint
print("Loaded checkpoint '{}' successfully.".format(resume_path010))

model_050 = CCPoint(args, backbone=DGCNN_cls(args)).to(device)
# checkpoint = torch.load(resume_path050, map_location='cpu')['point_model']
# checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
# model_050.load_state_dict(checkpoint, strict=True)
# del checkpoint
print("Loaded checkpoint '{}' successfully.".format(resume_path050))

model_100 = CCPoint(args, backbone=DGCNN_cls(args)).to(device)
checkpoint = torch.load(resume_path100, map_location='cpu')['point_model']
checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
model_100.load_state_dict(checkpoint, strict=True)
del checkpoint
print("Loaded checkpoint '{}' successfully.".format(resume_path100))

# model_000.eval()
model_010.eval()
model_050.eval()
model_100.eval()

feats_test_000 = []
feats_test_010 = []
feats_test_050 = []
feats_test_100 = []
labels_test_000 = []
labels_test_010 = []
labels_test_050 = []
labels_test_100 = []


for i, (data, label) in enumerate(test_val_loader):
    labels = list(map(lambda x: x[0], label.numpy().tolist()))
    data = data.to(device).permute(0, 2, 1)
    with torch.no_grad():
        # feats000 = model_000(data)[0]
        feats010 = model_010(data)[0]
        feats050 = model_050(data)[0]
        feats100 = model_100(data)[0]

    # for feat in feats000:
    #     feats_test_000.append(feat.cpu().numpy())

    for feat in feats010:
        feats_test_010.append(feat.cpu().numpy())

    for feat in feats050:
        feats_test_050.append(feat.cpu().numpy())

    for feat in feats100:
        feats_test_100.append(feat.cpu().numpy())

    # labels_test_000 += labels
    labels_test_010 += labels
    labels_test_050 += labels
    labels_test_100 += labels

# feats_test_000 = np.array(feats_test_000)
feats_test_010 = np.array(feats_test_010)
feats_test_050 = np.array(feats_test_050)
feats_test_100 = np.array(feats_test_100)

# feats_test_000_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(feats_test_000)
feats_test_010_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(feats_test_010)
feats_test_050_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(feats_test_050)
feats_test_100_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(feats_test_100)

# 创建自定义颜色映射
label_values = [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]
label_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
colors = sns.color_palette("hls", n_colors=len(label_values))
color_map = dict(zip(label_values, colors))

# sns.set_style('whitegrid')
# df_000 = pd.DataFrame()
# # "y" represents the number of samples
# df_000['label'] = labels_test_000
# df_000['axis-0'] = feats_test_000_embedded[:, 0]
# df_000['axis-1'] = feats_test_000_embedded[:, 1]
# figsize = (6, 5)
# fig, ax = plt.subplots(figsize=figsize)
# mn_ax = sns.scatterplot(ax=ax, x="axis-0", y="axis-1", hue=df_000.label.tolist(),
#                         # palette=sns.color_palette(palette='Set2', n_colors=40),
#                         # s=50,
#                         palette=color_map,
#                         data=df_000,
#                         legend=False)
# mn_ax.set(xlabel=None)
# mn_ax.set(ylabel=None)
# plt.grid(False)
# plt.show()
# res_fig = mn_ax.get_figure()
# res_fig.savefig('./output/visualization/tsne/model_000.png', dpi=300)

sns.set_style('whitegrid')
df_010 = pd.DataFrame()
# "y" represents the number of samples
df_010['label'] = labels_test_010
df_010['axis-0'] = feats_test_010_embedded[:, 0]
df_010['axis-1'] = feats_test_010_embedded[:, 1]
figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)
mn_ax = sns.scatterplot(ax=ax, x="axis-0", y="axis-1", hue=df_010.label.tolist(),
                        # palette=sns.color_palette(palette='Set2', n_colors=40),
                        # s=50,
                        palette=color_map,
                        data=df_010,
                        legend=False)
mn_ax.set(xlabel=None)
mn_ax.set(ylabel=None)
plt.grid(False)
plt.show()
res_fig = mn_ax.get_figure()
res_fig.savefig('./output/visualization/tsne/model_010.png', dpi=300)

sns.set_style('whitegrid')
df_050 = pd.DataFrame()
# "y" represents the number of samples
df_050['label'] = labels_test_050
df_050['axis-0'] = feats_test_050_embedded[:, 0]
df_050['axis-1'] = feats_test_050_embedded[:, 1]
figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)
mn_ax = sns.scatterplot(ax=ax, x="axis-0", y="axis-1", hue=df_050.label.tolist(),
                        # palette=sns.color_palette(palette='Set2', n_colors=40),
                        # s=50,
                        palette=color_map,
                        data=df_050,
                        legend=False)
mn_ax.set(xlabel=None)
mn_ax.set(ylabel=None)
plt.grid(False)
plt.show()
res_fig = mn_ax.get_figure()
res_fig.savefig('./output/visualization/tsne/model_050.png', dpi=300)

sns.set_style('whitegrid')
df_100 = pd.DataFrame()
# "y" represents the number of samples
df_100['label'] = labels_test_100
df_100['axis-0'] = feats_test_100_embedded[:, 0]
df_100['axis-1'] = feats_test_100_embedded[:, 1]
figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)
mn_ax = sns.scatterplot(ax=ax, x="axis-0", y="axis-1", hue=df_100.label.tolist(),
                        # palette=sns.color_palette(palette='Set2', n_colors=40),
                        # s=50,
                        # palette=sns.color_palette(palette="hls", n_colors=10),
                        palette=color_map,
                        data=df_100,
                        legend=False)
mn_ax.set(xlabel=None)
mn_ax.set(ylabel=None)
plt.grid(False)
plt.show()
res_fig = mn_ax.get_figure()
res_fig.savefig('./output/visualization/tsne/model_100.png', dpi=300)

# 创建颜色条
cmap = mcolors.ListedColormap(colors)
bounds = label_values
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 创建一个新的图形和轴
fig_legend, ax_legend = plt.subplots(figsize=(1, 5))

# 绘制圆点图例
for i, (label_name, label_value) in enumerate(zip(label_names, label_values)):
    color = color_map[label_value]
    ax_legend.scatter([0], [len(label_names) - 1 - i], color=color, label=label_name, s=100)
    ax_legend.text(0.05, len(label_names) - 1 - i, label_name, verticalalignment='center')
ax_legend.axis('off')
fig_legend.savefig('./output/visualization/tsne/legend.png', dpi=300, bbox_inches='tight')
plt.show()
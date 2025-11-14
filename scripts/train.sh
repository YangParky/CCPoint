#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE

# Pretrain
CUDA_VISIBLE_DEVICES=0,1 python train.py --exp ccpoint --batch_size 32 --world_size 2 --corrupt_affine affine_r5 \
--corrupt_extra random_masking --extra_level -1 --model dgcnn_part

# Finetune
#CUDA_VISIBLE_DEVICES=1 python ./downstream/segmentation/main_partseg.py --exp ccpoint --batch_size 32 \
#--world_size 1 --pretrain_path ./output/pretrain/dgcnn_part/models/checkpoint_best.pth.tar \
#--master_port 12385 --scheduler Step


# Pretrain
#CUDA_VISIBLE_DEVICES=0,1 python train.py ccpoint --batch_size 64 --world_size 2 --corrupt_affine affine_r5 \
#--corrupt_extra random_masking --extra_level -1 --model dgcnn_sem

# Finetune
#CUDA_VISIBLE_DEVICES=0,1 python ./downstream/segmentation/main_semseg.py --exp ccpoint --batch_size 48 --num_points 4096 \
#--world_size 2 --model dgcnn_sem --k 20 --pretrain_path ./output/pretrain/dgcnn_sem/ccpoint/models/checkpoint_best.pth.tar \
#--dataset s3dis --master_port 12345 --epoch 100 --scheduler Step --use_sgd
#
#CUDA_VISIBLE_DEVICES=0,1 python ./downstream/segmentation/main_semseg.py --exp ccpoint --batch_size 24 --num_points 4096 \
#--world_size 2 --model dgcnn_sem --k 20 --pretrain_path ./output/pretrain/dgcnn_sem/ccpoint/models/checkpoint_best.pth.tar \
#--dataset s3dis --master_port 12345 --epoch 100 --scheduler Cos


#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine None \
# --corrupt_extra None

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra None

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine None \
# --corrupt_extra random_noise

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine None \
# --corrupt_extra random_masking

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_masking

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_noise

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_mixing

# PointNet
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine None \
# --corrupt_extra None --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra None --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine None \
# --corrupt_extra random_noise --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine None \
# --corrupt_extra random_masking --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_noise --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_masking --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --corrupt_extra random_masking --model random_mixing

# corruption----DGCNN
#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine translate --corrupt_extra None \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine scale --corrupt_extra None \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine rotate --corrupt_extra None \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine reflection --corrupt_extra None \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine shear --corrupt_extra None \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine None --corrupt_extra add_global \
# -neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine None  --corrupt_extra add_local \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine None  --corrupt_extra dropout_global \
# --neg_lambda 0.2 --world_size 2

#CUDA_VISIBLE_DEVICES=0,1 python train.py --batch_size 32 --corrupt_affine None --corrupt_extra dropout_local \
# --neg_lambda 0.2 --world_size 2

# corruption----PointNet
#CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --corrupt_affine translate --corrupt_extra None --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --corrupt_affine scale --corrupt_extra None --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --corrupt_affine rotate --corrupt_extra None --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 64 --corrupt_affine reflection --corrupt_extra None --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 64 --corrupt_affine shear --corrupt_extra None --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 64 --corrupt_affine None --corrupt_extra add_global --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 64 --corrupt_affine None  --corrupt_extra add_local --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 64 --corrupt_affine None  --corrupt_extra dropout_global --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 64 --corrupt_affine None --corrupt_extra  dropout_local --neg_lambda 0.2 \
#--world_size 1 --model pointnet_cls


# affine level----DGCNN
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 0 --corrupt_extra random_masking --extra_level 0

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 1 --corrupt_extra random_masking --extra_level 1

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 2 --corrupt_extra random_masking --extra_level 2

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 3 --corrupt_extra random_masking --extra_level 3

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 64 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 4 --corrupt_extra random_masking --extra_level 4


# affine level----PointNet
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 0 --corrupt_extra random_masking --extra_level 0 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 1 --corrupt_extra random_masking --extra_level 1 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 2 --corrupt_extra random_masking --extra_level 2 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 3 --corrupt_extra random_masking --extra_level 3 --model pointnet_cls

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batch_size 96 --world_size 4 --corrupt_affine affine_r5 \
# --affine_level 4 --corrupt_extra random_masking --extra_level 4 --model pointnet_cls

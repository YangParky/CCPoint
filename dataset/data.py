import os
import glob
import h5py
import math
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from dataset.plyfile import load_ply
from dataset import data_utils as d_utils
from dataset import corrupt_utils as c_utils


ImageFile.LOAD_TRUNCATED_IMAGES = True

trans_pc_1 = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudNormalize(),
    ]
)
    
trans_pc_2 = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudNormalize(),
        d_utils.PointcloudScale(lo=0.5, hi=2, p=0.5),
        # d_utils.PointcloudHorizontalFlip(p=0.5),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudJitter(p=0.5),
        d_utils.PointcloudTranslate(0.5, p=0.5),
        d_utils.PointcloudRandomInputDropout(p=0.5),
        # d_utils.PointcloudRandomCutout(p=0.5),
        # d_utils.PointcloudRandomCrop(p=0.5),
    ]
)

trans_img_1 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
]
)

trans_img_2 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
]
)

corruptions = {
    'translate': c_utils.corrupt_tranlate,
    'scale': c_utils.corrupt_scale,
    'rotate': c_utils.corrupt_rotate,
    'reflection': c_utils.corrupt_reflection,
    'shear': c_utils.corrupt_shear,
    'jitter': c_utils.corrupt_jitter,
    'add_global': c_utils.corrupt_add_global,
    'add_local': c_utils.corrupt_add_local,
    'dropout_global': c_utils.corrupt_dropout_global,
    'dropout_local': c_utils.corrupt_dropout_local,
    'cut_out': c_utils.corrupt_cutout,
}

affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear', 'jitter']


def corrupt_data(data_instance, corrupt_type='affine_r3', affine_level=-1):
    # data_instance: N * 3
    if corrupt_type == 'affine_r5':
        number = random.choice([1, 2, 3, 4, 5, 6])
        adopted_affine = random.sample(affine_corruptions, number)
        for affine_corruption_item in adopted_affine:
            if affine_level != -1:
                level = affine_level
            else:
                level = random.choice([0, 1, 2, 3, 4])
            data_instance = corruptions[affine_corruption_item](data_instance, level)
    elif corrupt_type == 'affine_r3':
        number = random.choice([1, 2, 3])
        adopted_affine = random.sample(affine_corruptions, number)
        for affine_corruption_item in adopted_affine:
            if affine_level != -1:
                level = affine_level
            else:
                level = random.choice([0, 1, 2, 3, 4])
            data_instance = corruptions[affine_corruption_item](data_instance, level)
    else:
        pass

    return data_instance


def corrupt_single_func_data(data_instance, corrupt_type='translate', affine_level=-1):
    # data_instance: N * 3
    if corrupt_type != 'None':
        if affine_level != -1:
            level = affine_level
        else:
            level = random.choice([0, 1, 2, 3, 4])
        data_instance = corruptions[corrupt_type](data_instance, level)
    elif corrupt_type == 'None':
        pass
    else:
        raise Exception('Not implemented')

    return data_instance


def load_modelnet_data(partition, cat=40):

    # DATA_DIR = '/mnt/sda/xxy/Dataset/ModelNet40'
    DATA_DIR = '../../Dataset/ModelNet40'

    all_data, new_all_data = [],[]
    all_label, new_all_label= [],[]
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    if (cat == 10):
        for i in range(len(all_label)):
            if all_label[i] in [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]:
                # bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet
                new_all_data.append(all_data[i])
                new_all_label.append(all_label[i])
        all_data = np.array(new_all_data)
        all_label = np.array(new_all_label)
    return all_data, all_label


def load_ScanObjectNN(partition):
    # DATA_DIR = '/mnt/sda/xxy/Dataset/ScanObjectNN/h5_files/main_split'
    DATA_DIR = '../../Dataset/ScanObjectNN/h5_files/main_split'

    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label


def load_shapenet_data():
    # DATA_DIR = '/mnt/sda/xxy/Dataset/Shapenet'
    DATA_DIR = '../../Dataset/Shapenet'

    all_filepath = []

    # print('-'*5, 'IN load_shapenet_data()')
    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs

    return all_filepath


def get_render_imgs(pcd_path):
    path_lst = pcd_path.split('/')

    path_lst[-3] = 'ShapeNetRendering'
    path_lst[-1] = path_lst[-1][:-4]
    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))

    return img_path_list


class ShapeNetRender(Dataset):
    def __init__(self, n_imgs=1):
        self.data = load_shapenet_data()
        self.n_imgs = n_imgs

    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.sample(get_render_imgs(pcd_path), self.n_imgs)

        # for render_img_path in render_img_path_list:
        render_img_path1 = Image.open(render_img_path[0]).convert('RGB')
        render_img_path2 = Image.open(render_img_path[1]).convert('RGB')

        render_img1 = trans_img_1(render_img_path1)
        render_img2 = trans_img_2(render_img_path2)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_pc_1(pointcloud_1)
        point_t2 = trans_pc_2(pointcloud_2)

        pointclouds = (point_t1, point_t2)
        render_imgs = (render_img1, render_img2)

        return pointclouds, render_imgs

    def __len__(self):
        return len(self.data)


class ShapeNetRender_Corruption(Dataset):
    def __init__(self, corrupt_type='affine_r3', affine_level=5):
        self.data = load_shapenet_data()
        self.corrupt_type = corrupt_type
        self.affine_level = affine_level

    def __getitem__(self, item):
        pointcloud = load_ply(self.data[item])

        clean_point1 = trans_pc_1(pointcloud)
        clean_point2 = trans_pc_2(pointcloud)
        corruptpoint = trans_pc_1(corrupt_data(pointcloud, self.corrupt_type, self.affine_level))
        # corruptpoint = trans_pc_1(corrupt_single_func_data(pointcloud, self.corrupt_type, self.affine_level))

        return (clean_point1, clean_point2, corruptpoint)

    def __len__(self):
        return len(self.data)



class ShapeNetRender_Corruption_Single(Dataset):
    def __init__(self, corrupt_type='affine_r3', affine_level=5):
        self.data = load_shapenet_data()
        self.corrupt_type = corrupt_type
        self.affine_level = affine_level

    def __getitem__(self, item):
        pointcloud = load_ply(self.data[item])

        clean_point1 = trans_pc_1(pointcloud)
        clean_point2 = trans_pc_2(pointcloud)
        corruptpoint = corrupt_single_func_data(pointcloud, self.corrupt_type, self.affine_level)

        return (clean_point1, clean_point2, corruptpoint)

    def __len__(self):
        return len(self.data)


class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train', cat = 40):
        self.data, self.label = load_modelnet_data(partition, cat)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # shape_part = load_shapenet_data()
    # shape_part = get_render_imgs('/mnt/sda/xxy/Dataset/Shapenet/ShapeNet/02828884/dd56a9259a0eb458f155d75bbf62b80.ply')
    shape_part = ShapeNetRender_Corruption(n_imgs=2)
    train_loader = DataLoader(shape_part,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True
                              )
    clean_point1, clean_point2, corrput_point = shape_part.__getitem__(2)[0]
    for (point_t1, point_t2, corrupt_point), (image_t1, image_t2) in train_loader:
        print(point_t1.shape, corrupt_point.shape)
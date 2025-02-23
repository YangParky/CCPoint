import os

import torch
import math
import random
import numpy as np
import pandas as pd

from pyntcloud import PyntCloud


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    centroid = torch.mean(xyz, dim=1, keepdim=True)  # [B, 1, C]
    dist = torch.sum((xyz - centroid) ** 2, -1)
    farthest = torch.max(dist, -1)[1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).float()
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(pointcloud, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # print(idx.shape)
    device = pointcloud.device
    B = pointcloud.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = pointcloud[batch_indices, idx, :]
    return new_points


def global_transform(pointcloud, npoints):
    # Points: B N C
    device = pointcloud.device
    # points = points.permute(0, 2, 1)
    idx = farthest_point_sample(pointcloud, npoints)  # input BNC
    centroids = index_points(pointcloud, idx)   #[B, S, C]
    # U, S, V = batch_svd(centroids)
    U, S, V = torch.svd(pointcloud)
    # if train == True:
    #     index = torch.randint(2, (points.size(0), 1, 3)).type(torch.FloatTensor).cuda()
    #     V_ = V * index
    #     V -= 2 * V_
    # else:
    key_p = centroids[:, 0, :].unsqueeze(1)
    angle = torch.matmul(key_p, V)
    index = torch.le(angle, 0).type(torch.FloatTensor).to(device)
    V_ = V * index
    V -= 2 * V_
    # print(V.size()) ## 1 * 3 * 3
    xyz = torch.matmul(pointcloud, V)  #.permute(0, 2, 1)
    return xyz


def _pc_normalize(pc):
    """
    Normalize the point cloud to a unit sphere
    :param pc: input point cloud
    :return: normalized point cloud
    """
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    pc = pc / m
    return pc


def random_sample(pointcloud, num):
    ## input should be numpy arrays.
    if pointcloud.shape[0] >= num:
        permutation = np.arange(pointcloud.shape[0])
        np.random.shuffle(permutation)
        pointcloud = pointcloud[permutation[:num]]
    else:
        gap = num - pointcloud.shape[0]
        indices = np.random.choice(pointcloud.shape[0], gap, replace=True)
        pointcloud = np.vstack((pointcloud, pointcloud[indices]))
        permutation = np.arange(pointcloud.shape[0])
        np.random.shuffle(permutation)
        pointcloud = pointcloud[permutation[:num]]
    return pointcloud


def _shuffle_pointcloud(pointcloud):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(1, pointcloud.shape[1], 1).argsort(axis=1)
    return np.take_along_axis(pointcloud, idx, axis=1)


def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list


def _sample_points_inside_unit_sphere(batch_size, number_of_particles):
    """
    Uniformly sample points in a unit sphere
    :param number_of_particles: number of points to sample
    :return: sampled points
    """
    radius = np.random.uniform(0.0, 1.0, (batch_size, number_of_particles, 1))
    radius = np.power(radius, 1 / 3)
    costheta = np.random.uniform(-1.0, 1.0, (batch_size, number_of_particles, 1))
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, (batch_size, number_of_particles, 1))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.concatenate([x, y, z], axis=-1)


def corrupt_tranlate(pointcloud, level=4):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    xyz = np.random.uniform(low=-s, high=s, size=[3])
    return (pointcloud + xyz).astype('float32')


def corrupt_scale(pointcloud, level=4):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return np.multiply(pointcloud, xyz).astype('float32')


def corrupt_rotate(pointcloud, level=4):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi / 6
    angle_clip = angle_clip / 5 * (level + 1)
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    return np.dot(pointcloud, R)


def corrupt_reflection(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    choice = np.array([1, -1])
    reflection = np.random.choice(choice, size=(3))
    Rx = np.array([[reflection[0], 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    Ry = np.array([[1, 0, 0],
                   [0, reflection[1], 0],
                   [0, 0, 1]])
    Rz = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, reflection[2]]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(pointcloud, R)


def corrupt_shear(pointcloud, level=4):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)


def corrupt_jitter(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.01 * (level + 1)
    # B, N, C = pointcloud.shape
    # pointcloud = pointcloud + sigma * np.random.randn(B, N, C)

    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)

    return pointcloud.astype('float32')


def corrupt_add_global(pointcloud, level=None):
    """
    Add random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """

    npoints = 10 * (level + 1)
    additional_pointcloud = _sample_points_inside_unit_sphere(pointcloud.shape[0], npoints)
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:npoints, :]], axis=1)

    return torch.from_numpy(pointcloud).float()


def corrupt_add_local(pointcloud, level=None):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """

    # pointcloud_clone = pointcloud.cpu().numpy()
    pointcloud_clone = pointcloud

    num_points = pointcloud_clone.shape[1]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    pointcloud_clone = _shuffle_pointcloud(pointcloud_clone)
    add_pcd = np.zeros_like(pointcloud_clone)
    num_added = 0
    for i in range(num_clusters):
        K = cluster_size_list[i]
        sigma = np.random.uniform(0.075, 0.125)
        add_pcd[:, num_added:num_added+K, :] = np.copy(pointcloud_clone[:, i:i+1, :])
        add_pcd[:, num_added:num_added+K, :] = add_pcd[:, num_added:num_added+K, :] + sigma * np.random.randn(
            *add_pcd[:, num_added:num_added+K, :].shape)
        num_added += K
    assert num_added == total_cluster_size
    dist = np.sum(add_pcd ** 2, axis=-1, keepdims=True).repeat(3, axis=-1)
    add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    pointcloud_clone = np.concatenate([pointcloud_clone, add_pcd], axis=1)
    pointcloud_clone = pointcloud_clone[:, :num_points+total_cluster_size, :]

    return torch.from_numpy(pointcloud_clone).float()


def corrupt_dropout_global(pointcloud, level=None):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    num_points = pointcloud.shape[1]
    pointcloud = _shuffle_pointcloud(pointcloud)
    pointcloud = pointcloud[:, :int(num_points * (1 - drop_rate)), :]

    return pointcloud


def corrupt_dropout_local(pointcloud, level=None):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # pointcloud_clone = pointcloud.cpu().numpy()
    pointcloud_clone = pointcloud

    num_points = pointcloud_clone.shape[1]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud_clone = _shuffle_pointcloud(pointcloud_clone)
        dist = np.sum((pointcloud_clone - pointcloud_clone[:, :1, :]) ** 2, axis=-1, keepdims=True)
        idx = dist.argsort(axis=1)[:, ::-1, :]
        pointcloud_clone = np.take_along_axis(pointcloud_clone, idx, axis=1)
        num_points -= K
        pointcloud_clone = pointcloud_clone[:, :num_points, :]

    return torch.from_numpy(pointcloud_clone).float()


def corrupt_cutout(pointcloud, level=4):
    # pointcloud_clone = pointcloud.cpu().numpy()
    pointcloud_clone = pointcloud

    B, N, C = pointcloud_clone.shape
    c = [(2, 30), (3, 30), (5, 30), (7, 30), (10, 30)][level]

    for _ in range(c[0]):
        i = np.random.choice(N, 1)
        picked = pointcloud_clone[:, i, :]
        dist = np.sum((pointcloud_clone - picked) ** 2, axis=2, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=1)[:, :c[1]]

        mask = np.ones((B, N), dtype=bool)
        for b in range(B):
            mask[b, idx[b]] = False

        pointcloud_clone = pointcloud_clone[mask].reshape(B, -1, C)
        N = pointcloud_clone.shape[1]  # Update N after each cutout

    return torch.from_numpy(pointcloud_clone).float()

data_path = '/mnt/sda/xxy/Dataset/Shapenet/ShapeNet55-34/shapenet_pc/'
file_path = '/mnt/sda/xxy/Project/CCPoint/output/visualization/'

# file_names = os.listdir(data_path)
# # file_names = random.choices(file_names, k=20)
#
# # 筛选出文件名包含 '02691156' 的文件
# filtered_file_names = [file_name for file_name in file_names if '02691156' in file_name]
# # 从筛选出的文件中随机选择20个
# file_names = random.sample(filtered_file_names, min(20, len(filtered_file_names)))
#
# for name in file_names:
#     # file_name = '02691156-1a32f10b20170883663e90eaf6b4ca52'
#     file_name = name
#
#     sam = data_path + file_name
#
#     input = torch.from_numpy(np.load(sam))[:, :3]
#     input = _pc_normalize(input).unsqueeze(0)  # 8192 * 3
#     input_vanilla = global_transform(input, 32)  ## good pose for visualization
#
#     # no corruption.
#     idx = farthest_point_sample(input_vanilla, 2048)  # input BNC
#     points = index_points(input_vanilla, idx)[0]  # [B, S, C]
#     d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
#     cloud = PyntCloud(pd.DataFrame(data=d))
#     save_name = 'nocorruption-' + file_name.replace('.npy', '')
#     save_name = file_path + 'corruption/' + save_name + '.ply'
#     cloud.to_file(save_name)
#     print('visual no corruption successfully. Points shape: %s' % (str(points.cpu().numpy().shape),))

    # # corrupt_reflection
    # input = np.array(input_vanilla.squeeze(0))
    # input = corrupt_reflection(input, 3)
    # points = random_sample(input, 1024)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'reflection-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual reflection successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_rotation
    # input = np.array(input_vanilla.squeeze(0))
    # input = corrupt_rotate(input, 3)
    # points = random_sample(input, 1024)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'rotation-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual rotation successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_scale
    # input = np.array(input_vanilla.squeeze(0))
    # input = corrupt_scale(input, 3)
    # points = random_sample(input, 1024)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'scale-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual scale successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_shear
    # input = np.array(input_vanilla.squeeze(0))
    # input = corrupt_shear(input, 3)
    # points = random_sample(input, 1024)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'shear-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual shear successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_translate
    # input = np.array(input_vanilla.squeeze(0))
    # input = corrupt_tranlate(input, 3)
    # points = random_sample(input, 1024)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'translate-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual translate successfully. Points shape: %s' % (str(points.shape),))

    # corrupt_affinity
    # affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
    # affine_function = {
    #     'translate': corrupt_tranlate,
    #     'scale': corrupt_scale,
    #     'rotate': corrupt_rotate,
    #     'reflection': corrupt_reflection,
    #     'shear': corrupt_shear,
    # }
    # number = random.choice([1, 2])
    # adopted_affine = random.sample(affine_corruptions, number)
    # input = np.array(input_vanilla.squeeze(0))
    # for affine_corruption_item in adopted_affine:
    #     input = affine_function[affine_corruption_item](input, 2)
    # points = random_sample(input, 2048)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'affinity-' + file_name.replace('.npy', '')
    # save_name = file_path + 'corruption/' + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual affinity successfully. Points shape: %s' % (str(points.shape),))

    # # corrupt_dropout_global
    # input = np.array(input_vanilla)
    # input = np.expand_dims(random_sample(input.squeeze(0), 1024), axis=0)
    # points = corrupt_add_global(input, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'add_global-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual dropout_global successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_add_local
    # input = np.array(input_vanilla)
    # input = np.expand_dims(random_sample(input.squeeze(0), 1024), axis=0)
    # points = corrupt_add_local(input, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'add_local-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual add_local successfully. Points shape: %s' % (str(points.shape),))
    #
    # # jitter
    # input = np.array(input_vanilla.squeeze(0))
    # input = random_sample(input, 1024)
    # points = corrupt_jitter(input, 3)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'jitter-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual jitter successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_dropout_local
    # input = np.array(input_vanilla)
    # input = np.expand_dims(random_sample(input.squeeze(0), 1024), axis=0)
    # points = corrupt_dropout_global(input, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'dropout_global-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual dropout_local successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_dropout_local
    # input = np.array(input_vanilla)
    # input = np.expand_dims(random_sample(input.squeeze(0), 1024), axis=0)
    # points = corrupt_dropout_local(input, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'dropout_local-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual dropout_local successfully. Points shape: %s' % (str(points.shape),))
    #
    # # corrupt_cutout
    # input = np.array(input_vanilla)
    # input = np.expand_dims(random_sample(input.squeeze(0), 1024), axis=0)
    # points = corrupt_cutout(input, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'cut_out-' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual cut_out successfully. Points shape: %s' % (str(points.shape),))
    #
    # # combine corrupt_affinity add noise
    # affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
    # affine_function = {
    #     'translate': corrupt_tranlate,
    #     'scale': corrupt_scale,
    #     'rotate': corrupt_rotate,
    #     'reflection': corrupt_reflection,
    #     'shear': corrupt_shear,
    # }
    # nonaff_function = {
    #     'add_global': corrupt_add_global,
    #     'add_local': corrupt_add_local,
    #     'dropout_global': corrupt_dropout_global,
    #     'dropout_local': corrupt_dropout_local,
    #     'cutout': corrupt_cutout,
    #     'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
    # }
    # nonaff_extra = random.choices(['add_global', 'add_local'])[0]
    # number = random.choice([1, 2, 3, 4, 5])
    # adopted_affine = random.sample(affine_corruptions, number)
    # input = np.array(input_vanilla.squeeze(0))
    # for affine_corruption_item in adopted_affine:
    #     input = affine_function[affine_corruption_item](input, 3)
    # points = random_sample(input, 1024)
    # points = np.expand_dims(points, axis=0)
    # points = nonaff_function[nonaff_extra](points, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'comb_affinity-add' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual combination affine + add successfully. Points shape: %s' % (str(points.shape),))
    #
    # # combine corrupt_affinity drop points
    # affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
    # affine_function = {
    #     'translate': corrupt_tranlate,
    #     'scale': corrupt_scale,
    #     'rotate': corrupt_rotate,
    #     'reflection': corrupt_reflection,
    #     'shear': corrupt_shear,
    # }
    # nonaff_function = {
    #     'add_global': corrupt_add_global,
    #     'add_local': corrupt_add_local,
    #     'dropout_global': corrupt_dropout_global,
    #     'dropout_local': corrupt_dropout_local,
    #     'cutout': corrupt_cutout,
    #     'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
    # }
    # nonaff_extra = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
    # number = random.choice([1, 2, 3, 4, 5])
    # adopted_affine = random.sample(affine_corruptions, number)
    # input = np.array(input_vanilla.squeeze(0))
    # for affine_corruption_item in adopted_affine:
    #     input = affine_function[affine_corruption_item](input, 3)
    # points = random_sample(input, 1024)
    # points = np.expand_dims(points, axis=0)
    # points = nonaff_function[nonaff_extra](points, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'comb_affinity-drop' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual combination affine + drop successfully. Points shape: %s' % (str(points.shape),))
    #
    # # combine corrupt_affinity add + drop points
    # affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
    # affine_function = {
    #     'translate': corrupt_tranlate,
    #     'scale': corrupt_scale,
    #     'rotate': corrupt_rotate,
    #     'reflection': corrupt_reflection,
    #     'shear': corrupt_shear,
    # }
    # nonaff_function = {
    #     'add_global': corrupt_add_global,
    #     'add_local': corrupt_add_local,
    #     'dropout_global': corrupt_dropout_global,
    #     'dropout_local': corrupt_dropout_local,
    #     'cutout': corrupt_cutout,
    #     'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
    # }
    # nonaff_extra1 = random.choices(['add_global', 'add_local'])[0]
    # nonaff_extra2 = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
    # number = random.choice([1, 2, 3, 4, 5])
    # adopted_affine = random.sample(affine_corruptions, number)
    # input = np.array(input_vanilla.squeeze(0))
    # for affine_corruption_item in adopted_affine:
    #     input = affine_function[affine_corruption_item](input, 3)
    # points = random_sample(input, 1024)
    # points = np.expand_dims(points, axis=0)
    # points = nonaff_function[nonaff_extra1](points, 3).cpu().numpy()
    # points = nonaff_function[nonaff_extra2](points, 3).squeeze(0)
    # # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    # # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    # # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    # cloud = PyntCloud(pd.DataFrame(data=d))
    # save_name = 'comb_affinity-add-drop' + file_name
    # save_name = file_path + save_name + '.ply'
    # cloud.to_file(save_name)
    # print('visual combination affine + add + drop successfully. Points shape: %s' % (str(points.shape),))

    # combine corrupt_affinity add + drop points
    #     affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
    #     affine_function = {
    #         'translate': corrupt_tranlate,
    #         'scale': corrupt_scale,
    #         'rotate': corrupt_rotate,
    #         'reflection': corrupt_reflection,
    #         'shear': corrupt_shear,
    #     }
    #     nonaff_function = {
    #         'add_global': corrupt_add_global,
    #         'add_local': corrupt_add_local,
    #         'dropout_global': corrupt_dropout_global,
    #         'dropout_local': corrupt_dropout_local,
    #         'cutout': corrupt_cutout,
    #         'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
    #     }
    #     nonaff_extra1 = random.choices(['add_global', 'add_local'])[0]
    #     nonaff_extra2 = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
    #     number = random.choice([1, 2, 3, 4, 5])
    #     adopted_affine = random.sample(affine_corruptions, number)
    #     input = np.array(input_vanilla.squeeze(0))
    #     for affine_corruption_item in adopted_affine:
    #         input = affine_function[affine_corruption_item](input, 4)
    #     points = random_sample(input, 2048)
    #     points = np.expand_dims(points, axis=0)
    #     points = nonaff_function[nonaff_extra2](points, 4)
    #     points = nonaff_function[nonaff_extra1](points, 4).squeeze(0)
    #     # input_unsquee = torch.from_numpy(input).unsqueeze(0)
    #     # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
    #     # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
    #     d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    #     cloud = PyntCloud(pd.DataFrame(data=d))
    #     save_name = 'comb_affinity-add-drop' + file_name.replace('.npy', '')
    #     save_name = file_path + 'corruption/' + save_name + '.ply'
    #     cloud.to_file(save_name)
    #     print('visual combination affine + drop + add successfully. Points shape: %s' % (str(points.shape),))

file_name = '02691156-1a32f10b20170883663e90eaf6b4ca52.npy'
sam = data_path + file_name
input = torch.from_numpy(np.load(sam))[:, :3]
input = _pc_normalize(input).unsqueeze(0)  # 8192 * 3
input_vanilla = global_transform(input, 32)  ## good pose for visualization

# # combine corrupt_affinity add + drop points
# affine_corruptions = ['translate', 'scale', 'rotate', 'reflection', 'shear']
# affine_function = {
#     'translate': corrupt_tranlate,
#     'scale': corrupt_scale,
#     'rotate': corrupt_rotate,
#     'reflection': corrupt_reflection,
#     'shear': corrupt_shear,
# }
# nonaff_function = {
#     'add_global': corrupt_add_global,
#     'add_local': corrupt_add_local,
#     'dropout_global': corrupt_dropout_global,
#     'dropout_local': corrupt_dropout_local,
#     'cutout': corrupt_cutout,
#     'None': lambda point_cr, extra_level: point_cr  # 如果不需要操作，直接返回原数据
# }
# nonaff_extra1 = random.choices(['add_global', 'add_local'])[0]
# nonaff_extra2 = random.choices(['dropout_global', 'dropout_local', 'cutout'])[0]
# number = random.choice([1, 2, 3, 4, 5])
# adopted_affine = random.sample(affine_corruptions, number)
# input = np.array(input_vanilla.squeeze(0))
# for affine_corruption_item in adopted_affine:
#     input = affine_function[affine_corruption_item](input, 0)
# points = random_sample(input, 1024)
# points = np.expand_dims(points, axis=0)
# points = nonaff_function[nonaff_extra2](points, 0)
# points = nonaff_function[nonaff_extra1](points, 0).squeeze(0)
# # input_unsquee = torch.from_numpy(input).unsqueeze(0)
# # idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# # points = index_points(input_unsquee, idx)[0]  # [B, S, C]
# d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
# cloud = PyntCloud(pd.DataFrame(data=d))
# save_name = '0comb_affinity-add-drop' + file_name.replace('.npy', '')
# save_name = file_path + 'corruption/' + save_name + '.ply'
# cloud.to_file(save_name)
# print('visual combination affine + drop + add successfully. Points shape: %s' % (str(points.shape),))

input = np.array(input_vanilla.squeeze(0))
points = random_sample(input, 256)
d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'reconstruction_256p' + file_name.replace('.npy', '')
save_name = file_path + 'corruption/' + save_name + '.ply'
cloud.to_file(save_name)
print('visual reconstruction 256 points successfully. Points shape: %s' % (str(points.shape),))
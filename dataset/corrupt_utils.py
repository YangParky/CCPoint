import numpy as np
import math
import torch


def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(1, pcd.shape[1], 1).argsort(axis=1)
    return np.take_along_axis(pcd, idx, axis=1)


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
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:, :npoints, :]], axis=1)

    return torch.from_numpy(pointcloud).float()


def corrupt_add_local(pointcloud, level=None):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """

    pointcloud_clone = pointcloud.cpu().numpy()

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

    return pointcloud.float()


def corrupt_dropout_local(pointcloud, level=None):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    pointcloud_clone = pointcloud.cpu().numpy()

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
    pointcloud_clone = pointcloud.cpu().numpy()

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

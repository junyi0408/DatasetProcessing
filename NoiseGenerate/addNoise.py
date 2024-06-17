import numpy as np
import open3d as o3d

# 高斯噪声是最常见的噪声类型，它按照高斯分布在点云的坐标上添加噪声。通常用于模拟传感器测量误差。
def add_gaussian_noise(pcd, mean=0, std=0.01):
    noise = np.random.normal(mean, std, size=np.asarray(pcd.points).shape)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise)
    return pcd


# 均匀噪声在一个给定范围内均匀分布，适用于模拟传感器系统中的均匀误差。
def add_uniform_noise(pcd, low=-0.01, high=0.01):
    noise = np.random.uniform(low, high, size=np.asarray(pcd.points).shape)
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + noise)
    return pcd


# 椒盐噪声会随机地将一些点的位置设为某个极端值，模拟一些传感器采样过程中出现的突发误差。
def add_salt_and_pepper_noise(pcd, salt_prob=0.01, pepper_prob=0.01, low=-0.1, high=0.1):
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    
    salt_num = int(num_points * salt_prob)
    pepper_num = int(num_points * pepper_prob)
    
    salt_indices = np.random.choice(num_points, salt_num, replace=False)
    pepper_indices = np.random.choice(num_points, pepper_num, replace=False)
    
    points[salt_indices] = np.random.uniform(high, high, points[salt_indices].shape)
    points[pepper_indices] = np.random.uniform(low, low, points[pepper_indices].shape)
    
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# 随机丢失会随机移除一些点，模拟传感器在某些区域无法获取数据的情况。
def add_random_dropout(pcd, dropout_ratio=0.01):
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    
    keep_ratio = 1 - dropout_ratio
    keep_num = int(num_points * keep_ratio)
    
    keep_indices = np.random.choice(num_points, keep_num, replace=False)
    
    pcd.points = o3d.utility.Vector3dVector(points[keep_indices])
    return pcd



# 局部扰动会在点云的某个局部区域添加噪声，模拟局部的测量误差。
def add_local_perturbation(pcd, center, radius, std=0.01):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - center, axis=1)
    
    perturb_indices = np.where(distances < radius)[0]
    perturb_noise = np.random.normal(0, std, size=(len(perturb_indices), 3))
    
    points[perturb_indices] += perturb_noise
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

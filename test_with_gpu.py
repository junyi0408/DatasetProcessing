import os
import shutil
import numpy as np
import open3d as o3d
import torch

source_file = "./02691156/"
destination_file = "./result/"
file_num = 100

def extract_model(source_file, destination_file, file_num):
    files = os.listdir(source_file)
    num_files = len(files)
    distance_matrix = torch.zeros((num_files, num_files), device=device)

    k = 0
    for i in range(num_files):
        mesh1 = o3d.io.read_triangle_mesh(os.path.join(source_file, files[i]))
        pcd1 = torch.tensor(np.asarray(mesh1.sample_points_uniformly(1000).points), device=device)

        for j in range(i + 1, num_files):
            mesh2 = o3d.io.read_triangle_mesh(os.path.join(source_file, files[j]))
            pcd2 = torch.tensor(np.asarray(mesh2.sample_points_uniformly(1000).points), device=device)

            distance_matrix[i, j] = distance_matrix[j, i] = chamfer_distance(pcd1, pcd2)
            k += 1
            print(k)
            if k % 100000 == 0:
                np.savetxt(f'./mat{int(k/100000)}.txt', distance_matrix.cpu().numpy(), fmt='%.8f')

    indexs = find_m_most_different_objects(distance_matrix.cpu().numpy(), file_num)
    for index in indexs:
        shutil.copy2(os.path.join(source_file, files[index]), destination_file)

def chamfer_distance(pcd1, pcd2):
    # d1 = torch.cdist(pcd1, pcd2).min(dim=1)[0]
    # d2 = torch.cdist(pcd2, pcd1).min(dim=1)[0]
    chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    points1 = pcd1.unsqueeze(0)
    points2 = pcd2.unsqueeze(0)
    dist1, dist2, _, _ = chamLoss(points1, points2)
    return torch.mean(dist1) + torch.mean(dist2)

def find_m_most_different_objects(D, m):
    n = D.shape[0]
    avg_distances = np.sum(D, axis=1) / (n - 1)
    i0 = np.argmax(avg_distances)
    S = {i0}

    while len(S) < m:
        max_distance_sum = -1
        candidate = -1
        for i in range(n):
            if i not in S:
                distance_sum = sum(D[i, j] for j in S)
                if distance_sum > max_distance_sum:
                    max_distance_sum = distance_sum
                    candidate = i
        S.add(candidate)
    
    return list(S)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extract_model(source_file, destination_file, file_num)

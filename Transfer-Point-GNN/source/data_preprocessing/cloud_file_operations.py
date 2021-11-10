import os
import pye57
from source.dataset_classes.point_cloud_processing import read_e57, Points
import numpy as np
import open3d
import plyfile


cloud_compare = 'C:\\Program Files\\CloudCompare'
path = 'F:\\TransferPoint-GNN\\MUT\\tunnel'

def analyze_e57_files_for_pye57_error(path):
    damaged_files = []
    for file in os.listdir(path):
        print(f"Analyzing {file}")
        try:
            e57 = pye57.E57(os.path.join(path, file))
            data = e57.read_scan_raw(0)
            print("  File good.")
        except pye57.libe57.E57Exception:
            damaged_files.append(file)
            print("  File bad.")
    return damaged_files

# check for splitted cloud extents ============================================
def analyze_extents_of_splitted_tunnel_clouds(splitted_tunnel_dir, target_file):
    with open(target_file, 'w') as file:
        min_xdiff = 100
        min_ydiff = 100
        min_zdiff = 100
        for cloud in os.listdir(splitted_tunnel_dir):
            cloud_path = os.path.join(splitted_tunnel_dir, cloud)
            points = read_e57(cloud_path)
            if points is None:
                continue
            mins = points.xyz.min(axis=0)
            maxs = points.xyz.max(axis=0)

            x_diff = maxs[0] - mins[0]
            y_diff = maxs[1] - mins[1]
            z_diff = maxs[2] - mins[2]

            diffs = {
                cloud: (x_diff, y_diff, z_diff)
            }

            min_xdiff = min(min_xdiff, x_diff)
            min_ydiff = min(min_ydiff, y_diff)
            min_zdiff = min(min_zdiff, z_diff)

            file.write(f'{diffs}\n')

        file.write(f'min_xdiff={min_xdiff}')
        file.write(f'min_ydiff={min_ydiff}')
        file.write(f'min_zdiff={min_zdiff}')


def crop_E57_with_CC():
    """ WARNING!!!: CC docs: each cloud will be replaced in memory by its cropped version
    (since version 2.11, the cloud is removed from memory if it's totally cropped out)"""

    os.system(f'cmd /c "cd {cloud_compare} &'
              f'CloudCompare -AUTO_SAVE {{OFF}}'
              f'CloudCompare --C_EXPORT_FMT {{E57}}')

    tunnel_dir = ''
    file = ''
    e57 = os.path.join(tunnel_dir, file)
    os.system(f'cmd /c "cd {cloud_compare} &'
              f'CloudCompare -O {e57} -CROP{{Xmin:Ymin:Zmin:Xmax:Ymax:Zmax}} -SAVE_CLOUDS FILE {1,2,3}"')

# PLY----------------------------------
def convert_e57_to_ply_with_CC():
    os.system(f'cmd /c "cd {cloud_compare} &'
              f'CloudCompare --C_EXPORT_FMT {{PLY}}')
    tunnel = 'F:\\TransferPoint-GNN\\MUT\\tunnel'
    tunnel_ply = 'F:\\TransferPoint-GNN\\MUT\\tunnel.ply'

    for file in os.listdir(tunnel):
        e57 = os.path.join(tunnel, file)
        file_name = file.split(".")[0]
        ply = os.path.join(tunnel_ply, f'{file_name}.ply')
        os.system(f'cmd /c "cd {cloud_compare} &'
                  f'CloudCompare -O {e57} -SAVE_CLOUDS FILE {ply}"')

def read_ply():
    point_file = "F:\\TransferPoint-GNN\\Section1.ply"
    point_file = "F:\\TransferPoint-GNN\\MUT\\tunnel-ply\\5-1_539400_539500.ply"

    data = plyfile.PlyData.read(point_file)
    temp = data['vertex']
    intensity = temp.data['scalar_Intensity']
    x = temp.data['x']
    y = temp.data['y']
    z = temp.data['z']

    velo_points = []
    for num_x, num_y, num_z in x, y, z:
        velo_points.append(num_x)
        velo_points.append(num_y)
        velo_points.append(num_z)
    velo_points = np.asarray(velo_points)


    pcd = open3d.io.read_point_cloud(point_file)
    downsampled_xyz = np.asarray(pcd.voxel_down_sample(voxel_size=0.8).points)
    print(downsampled_xyz)  # unshifted original coordinates

# BIN-----------------------------------
def read_bin():
    point_file = "F:\\TransferPoint-GNN\\KITTI\\velodyne\\testing\\velodyne\\000002.bin"
    #point_file = "F:\\TransferPoint-GNN\\000.bin"

    pcd_points = np.fromfile(point_file, dtype=np.float32)
    reshaped = pcd_points.reshape(-1, 4)
    velo_points = reshaped[:, :3]
    reflections = reshaped[:, [3]]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(velo_points)  # 59878339
    open3d.visualization.draw_geometries([pcd])
    downsampled_xyz = np.asarray(pcd.voxel_down_sample(voxel_size=0.8).points)  # Fehler
    return Points(velo_points, reflections)

import os
import pye57
import numpy as np
import open3d
import random
from collections import namedtuple

Points = namedtuple('Points', ['xyz', 'attr'])

def downsample_by_average_voxel(points, voxel_size):
    """Voxel downsampling using average function.

  points: a Points namedtuple containing "xyz" and "attr".
  voxel_size: the size of voxel cells used for downsampling.
  """
    # create voxel grid
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    xyz_idx = (points.xyz - xyz_offset) // voxel_size
    xyz_idx = xyz_idx.astype(np.int32)
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1] * dim_x + xyz_idx[:, 2] * dim_y * dim_x
    order = np.argsort(keys)
    keys = keys[order]
    points_xyz = points.xyz[order]
    unique_keys, lens = np.unique(keys, return_counts=True)
    indices = np.hstack([[0], lens[:-1]]).cumsum()
    downsampled_xyz = np.add.reduceat(
        points_xyz, indices, axis=0) / lens[:, np.newaxis]
    include_attr = points.attr is not None
    if include_attr:
        attr = points.attr[order]
        downsampled_attr = np.add.reduceat(
            attr, indices, axis=0) / lens[:, np.newaxis]
    if include_attr:
        return Points(xyz=downsampled_xyz,
                      attr=downsampled_attr)
    else:
        return Points(xyz=downsampled_xyz,
                      attr=None)


def downsample_by_random_voxel(points, voxel_size, add_rnd3d=False):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)

    if not add_rnd3d:
        xyz_idx = (points.xyz - xyz_offset) // voxel_size
    else:
        xyz_idx = (points.xyz - xyz_offset +
                   voxel_size * np.random.random((1, 3))) // voxel_size
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1] * dim_x + xyz_idx[:, 2] * dim_y * dim_x
    num_points = xyz_idx.shape[0]

    voxels_idx = {}
    for pidx in range(len(points.xyz)):
        key = keys[pidx]
        if key in voxels_idx:
            voxels_idx[key].append(pidx)
        else:
            voxels_idx[key] = [pidx]

    downsampled_xyz = []
    downsampled_attr = []
    for key in voxels_idx:
        center_idx = random.choice(voxels_idx[key])
        downsampled_xyz.append(points.xyz[center_idx])
        downsampled_attr.append(points.attr[center_idx])

    return Points(xyz=np.array(downsampled_xyz),
                  attr=np.array(downsampled_attr))

def display_cloud(cloud_points, window_name=''):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np.asarray(cloud_points))
    open3d.visualization.draw_geometries([cloud], window_name=window_name)

def display_cloud_by_filename(filename):
    points, trans = read_e57(filename)
    points_xyz = points.xyz
    display_cloud(points_xyz)

def read_e57(point_file):
    e57 = pye57.E57(point_file)
    header = e57.get_header(0)
    translation = header.translation
    data = e57.read_scan_raw(0)
    assert isinstance(data["cartesianX"], np.ndarray)
    assert isinstance(data["cartesianY"], np.ndarray)
    assert isinstance(data["cartesianZ"], np.ndarray)
    assert isinstance(data["intensity"], np.ndarray)
    refl_intensity = data["intensity"].reshape(-1,1)/data["intensity"].max()
    x = data["cartesianX"]
    y = data["cartesianY"]
    z = data["cartesianZ"]
    e57_points = np.vstack((y, z, x)).T

    return Points(xyz=e57_points, attr=refl_intensity), translation


def downsample_e57(points,
                   initial_downsample_size=0.07,
                   second_layer_downsample_size=0.25,
                   visualize=False):

    if initial_downsample_size is not None:
        points = downsample_by_random_voxel(points, initial_downsample_size)
        print(f'initial downsample yields: {points.xyz.shape}')

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.xyz)  # 59 878 280 points
    if visualize:
        open3d.visualization.draw_geometries([pcd])

    if second_layer_downsample_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=second_layer_downsample_size)  # 0.02 -> 7 859 105; 0.4 -> 36528
        #print(f'second downsample yields: {points.xyz.shape}')
    if visualize:
        open3d.visualization.draw_geometries([pcd])
    res = np.asarray(pcd.points)  # 0.1 -> 409 164
    return res
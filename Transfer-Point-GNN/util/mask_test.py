import os
from source.dataset_classes.dataset import Dataset
from source.dataset_classes.point_cloud_processing import downsample_e57

DATASET_DIR = "F:\\TransferPoint-GNN\\MUT_mini"
DATASET = 'MUT'
DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
                                  '3DOP_splits', 'train_rot_sym.txt')
if DATASET == 'KITTI':
    image_path = 'image/training/image_2'
    point_cloud_dir = 'velodyne/training/velodyne/'
    calib_path = 'calib/training/calib/'
    label_path = 'labels/training/label_2'
    point_cloud_format = '.bin'
    is_raw = False
elif DATASET == 'MUT':  # changed: variable paths
    image_path = ''
    point_cloud_dir = 'tunnel'
    calib_path = ''
    label_path = 'rot_sym_labels'
    point_cloud_format = '.e57'
    is_raw = True

dataset = Dataset(
    os.path.join(DATASET_DIR, image_path),
    os.path.join(DATASET_DIR, point_cloud_dir),
    os.path.join(DATASET_DIR, calib_path),
    os.path.join(DATASET_DIR, label_path),
    point_cloud_format,
    DATASET_SPLIT_FILE,
    num_classes=4,
    is_raw=is_raw)

label_method = "posts_rot"

e57_file = f'F:\\TransferPoint-GNN\\MUT_mini\\tunnel\\5-1_539000_539100(3).e57'
last_layer_points = dataset.read_e57_by_frame_id(3)
last_layer_points_xyz = downsample_e57(last_layer_points,
                                       initial_downsample_size=0.07,
                                       second_layer_downsample_size=0.2, visualize=False)

# 0 -> 11, 1 -> 13, 2 -> 15
# 3 -> 3, 4 -> 4, 5 -> 6, 6 -> 8
box_label_list = dataset.get_label(3)

cls_labels, boxes_3d, valid_boxes, label_map = \
    dataset.assign_classaware_label_to_points(box_label_list, last_layer_points_xyz, label_method, expend_factor=(1.0, 1.0, 1.0))

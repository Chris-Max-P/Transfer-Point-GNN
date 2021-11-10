import pathlib

import tensorflow as tf
import os

from source.dataset_classes.dataset import get_input_features, assign_classaware_label_to_points, get_label
from models.box_encoding import get_box_encoding_fn
from models.graph_gen import get_graph_generate_fn
from util.paths_and_data import tunnel_dir
from source.dataset_classes.point_cloud_processing import read_e57, downsample_by_random_voxel
from util_point_gnn.config_util import load_config
import numpy as np
import pye57

#tunnel_dir = tunnel_dir.replace('MUT', 'MUT_mini')
config_path = "configs\\posts_rot_1024_1_config"
meta_config_path = "configs\\posts_rot_1024_1_meta_config"
config = load_config(config_path)
meta_config = load_config(meta_config_path)
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
batch_size = 1
DATASET_ROOT_DIR = ""
split_path = os.path.join(DATASET_ROOT_DIR, f"3DOP_splits\\{meta_config['train_dataset']}")

#tf.compat.v1.disable_eager_execution()

# get dataset from split file ==============================
def get_file_name_tensors_from_split(split_path):
    with open(split_path) as split_file:
        split_clouds = split_file.readlines()
        file_name_tensors = []
        for file_name in split_clouds:
            file_name_tensors.append(tf.convert_to_tensor(os.path.join(tunnel_dir, file_name.replace('txt\n', 'e57'))))
    return file_name_tensors


def filter_dataset_with_split(dataset, file_name_tensors):
    new_dataset = []
    for el in dataset:
        if el in file_name_tensors:
            new_dataset.append(el)
    dataset = tf.data.Dataset.from_tensor_slices(new_dataset)
    return dataset
#dataset = tf.compat.v1.data.Dataset.list_files(f'{tunnel_dir}/*.e57')
#dataset = filter_dataset_with_split(dataset, file_name_tensors)

def get_list(tup):
    values, row_split = tup
    tf.RaggedTensor.from_row_splits(values, row_split)
    return list(tf.RaggedTensor.from_row_splits(values, row_split))

file_name_tensors = get_file_name_tensors_from_split()
dataset = tf.compat.v1.data.Dataset.from_tensor_slices(file_name_tensors)

# get data from each file ===================================
def fetch_dataV2(tunnel_path):
    tunnel_path = tunnel_path.numpy().decode('utf-8')

    cloud_points, translation = read_e57(tunnel_path)
    if config['downsample_by_voxel_size'] is not None:
        cloud_points_first_level = downsample_by_random_voxel(cloud_points, config['downsample_by_voxel_size'])

    tunnel_path = pathlib.Path(tunnel_path)
    label_path = tunnel_path.parent.parent.joinpath('rot_sym_labels', tunnel_path.with_suffix('.txt').name)
    box_label_list = get_label(label_path)

    graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cloud_points_first_level.xyz, **config['graph_gen_kwargs'])
    input_v = get_input_features(cloud_points_first_level, config['input_features'])

    last_layer_graph_level = \
        config['model_kwargs']['layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]
    label_method = config['label_method']

    cls_labels, boxes_3d, valid_boxes, label_map = \
        assign_classaware_label_to_points(box_label_list, last_layer_points_xyz, label_method,
                                          expend_factor=meta_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz,
                                    boxes_3d, label_map)

    input_v = tf.constant(input_v.astype(np.float32))

    vertex_coord_list = tf.ragged.stack([tf.constant(p.astype(np.float32)) for p in vertex_coord_list])
    keypoint_indices_list = tf.ragged.stack([tf.constant(e.astype(np.int32)) for e in keypoint_indices_list])
    edges_list = tf.ragged.stack([tf.constant(e.astype(np.int32)) for e in edges_list])
    cls_labels = tf.constant(cls_labels.astype(np.int32))
    encoded_boxes = tf.constant(encoded_boxes.astype(np.float32))
    valid_boxes = tf.constant(valid_boxes.astype(np.float32))
    return (input_v,
            vertex_coord_list.values.to_tensor(), vertex_coord_list.row_splits,
            keypoint_indices_list.values.to_tensor(), keypoint_indices_list.row_splits,
            edges_list.values.to_tensor(), edges_list.row_splits,
            cls_labels,
            encoded_boxes,
            valid_boxes)

dataset = dataset.map(lambda x: tf.py_function(func=fetch_dataV2, inp=[x],
                                               Tout=(tf.float32,
                                                           tf.float32, tf.int64,
                                                           tf.int32, tf.int64,
                                                           tf.int32, tf.int64,
                                                           tf.int32,
                                                           tf.float32,
                                                           tf.float32)))

current_dataset = dataset.take(batch_size)
for batch in current_dataset:
    print(batch)
    input_v = batch[0]
    vertex_coord_list = get_list(batch[1:3])
    keypoint_indices_list = get_list(batch[3:5])
    edges_list = get_list(batch[5:7])
    cls_labels = batch[7]
    encoded_boxes = batch[8]
    valid_boxes = batch[9]


#el_spec = cloud_data.element_spec
#cloud_data = cloud_data.cache("cache")

    #data_list = list(cloud_data.as_numpy_iterator()) #%%


# list files
# map data
# cache

#===> shuffle
# repeat -> num epochs => einfügen statt for batch_idx in range ...
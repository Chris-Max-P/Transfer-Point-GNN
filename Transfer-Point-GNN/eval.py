"""This file defines the evaluation process of Point-GNN object detection."""

import os
import shutil
import time
import argparse
import zipfile

import numpy as np
import tensorflow as tf

from eval_mut.statistics import append_epoch_to_stats_dict, evaluate_json_statistics, \
    initialize_stats_dict
from source.dataset_classes.dataset import create_dataset, get_input_features
from source.dataset_classes.point_cloud_processing import downsample_by_random_voxel
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
    get_encoding_len
from models import preprocess
from util_point_gnn.config_util import load_train_config, save_config
from util_point_gnn.summary_util import write_summary_scale

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

parser = argparse.ArgumentParser(description='Repeated evaluation of Trans-PointGNN.')
parser.add_argument('checkpoint', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                    help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--label_dir', type=str, default='labels',
                    help='only for MUT: rot_sym_labels as second option')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--eval_while_training', type=bool, default=True)

args = parser.parse_args()

checkpoint = args.checkpoint
DATASET = args.dataset
LABEL_DIR = args.label_dir
DATASET_DIR = args.dataset_root_dir
np_preprocessed_dir = os.path.join(DATASET_DIR, 'np_preprocessed')

if DATASET == 'MUT':
    properties = checkpoint.split('_')
    config_name = ''
    keep = 4
    for x in range(keep):
        config_name += properties[x] + '_'

    checkpoint_path = f'./checkpoints/{checkpoint}'
    config_path = f'./configs/{config_name}config'
    meta_config_path = f'./configs/{checkpoint}_meta_config'

elif DATASET == 'KITTI':
    checkpoint_path = f'./checkpoints/{checkpoint}'
    config_path = f'./configs/{checkpoint}_config'
    meta_config_path = f'./configs/{checkpoint}_meta_config'

config = load_train_config(config_path)
meta_config = load_train_config(meta_config_path)

split = args.split
eval_while_training_on = args.eval_while_training

DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
                                  './3DOP_splits/' + meta_config[f'{split}_dataset'])

if 'eval' in config:
    config = config['eval']

dataset = create_dataset(DATASET, LABEL_DIR, DATASET_DIR, DATASET_SPLIT_FILE, config)
NUM_CLASSES = dataset.num_classes

NUM_TEST_SAMPLE = dataset.num_files

BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])


def fetch_data(frame_idx):

    np_path = os.path.join(np_preprocessed_dir, f'{dataset.get_filename(frame_idx)}.npz')
    if os.path.exists(np_path):
        data = np.load(np_path, allow_pickle=True)
        try:
            input_v = data['input_v']
            vertex_coord_list = data['vertex_coord_list']
            keypoint_indices_list = data['keypoint_indices_list']
            edges_list = data['edges_list']
            cls_labels = data['cls_labels']
            encoded_boxes = data['encoded_boxes']
            valid_boxes = data['valid_boxes']

            return (input_v, vertex_coord_list, keypoint_indices_list, edges_list,
                    cls_labels, encoded_boxes, valid_boxes)
        except zipfile.BadZipfile:
            pass
    if DATASET == 'MUT':
        cam_rgb_points = dataset.get_velo_points(frame_idx)
        if config['downsample_by_voxel_size'] is not None:
            cam_rgb_points = downsample_by_random_voxel(cam_rgb_points, config['downsample_by_voxel_size'])

    elif DATASET == 'KITTI':
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx,
                                                                  config['downsample_by_voxel_size'])

    box_label_list = dataset.get_label(frame_idx)
    graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
    (vertex_coord_list, keypoint_indices_list, edges_list) = \
        graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])
    input_v = get_input_features(cam_rgb_points, config['input_features'])

    last_layer_graph_level = \
        config['model_kwargs']['layer_configs'][-1]['graph_level']
    last_layer_points_xyz = vertex_coord_list[last_layer_graph_level + 1]
    label_method = config['label_method']

    cls_labels, boxes_3d, valid_boxes, label_map = \
        dataset.assign_classaware_label_to_points(box_label_list, last_layer_points_xyz, label_method,
                                                  expend_factor=meta_config.get('expend_factor', (1.0, 1.0, 1.0)))
    encoded_boxes = box_encoding_fn(cls_labels, last_layer_points_xyz, boxes_3d, label_map)

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]
    cls_labels = cls_labels.astype(np.int32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)

    np.savez(np_path,
             input_v=input_v, vertex_coord_list=vertex_coord_list,
             keypoint_indices_list=keypoint_indices_list,
             edges_list=edges_list, cls_labels=cls_labels,
             encoded_boxes=encoded_boxes, valid_boxes=valid_boxes)

    return (input_v, vertex_coord_list, keypoint_indices_list, edges_list,
            cls_labels, encoded_boxes, valid_boxes)


# model =======================================================================
tf.compat.v1.disable_eager_execution()

def get_t_initial_vertex_features(config_input_features):
    if config_input_features == 'irgb':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 4])
    elif config_input_features == 'rgb':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 3])
    elif config_input_features == '0000':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 4])
    elif config_input_features == 'i000':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 4])
    elif config_input_features == 'i':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 1])
    elif config_input_features == '0':
        t_initial_vertex_features = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, 1])
    return t_initial_vertex_features


t_initial_vertex_features = get_t_initial_vertex_features(config['input_features'])

t_vertex_coord_list = [tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3])]
t_edges_list = []
t_keypoint_indices_list = []

for _ in range(len(config['graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3]))
    t_edges_list.append(
        tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2]))
    t_keypoint_indices_list.append(
        tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1]))

t_class_labels = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])
t_encoded_gt_boxes = tf.compat.v1.placeholder(
    dtype=tf.float32, shape=[None, 1, BOX_ENCODING_LEN])
t_valid_gt_boxes = tf.compat.v1.placeholder(
    dtype=tf.float32, shape=[None, 1, 1])
t_is_training = tf.compat.v1.placeholder(dtype=tf.bool, shape=[])

model = get_model(config['model_name'])(num_classes=NUM_CLASSES,
                                        box_encoding_len=BOX_ENCODING_LEN, mode='eval',
                                        **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list,
    t_keypoint_indices_list, t_edges_list, t_is_training)
t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
t_loss_dict = model.loss(t_logits, t_class_labels, t_pred_box,
                         t_encoded_gt_boxes, t_valid_gt_boxes, **config['loss'])
t_cls_loss = t_loss_dict['cls_loss']
t_loc_loss = t_loss_dict['loc_loss']
t_reg_loss = t_loss_dict['reg_loss']
t_classwise_loc_loss = t_loss_dict['classwise_loc_loss']
t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss

t_classwise_loc_loss_update_ops = {}
for class_idx in range(NUM_CLASSES):
    for bi in range(BOX_ENCODING_LEN):
        classwise_loc_loss_ind = t_classwise_loc_loss[class_idx][bi]
        t_mean_loss, t_mean_loss_op = tf.compat.v1.metrics.mean(
            classwise_loc_loss_ind,
            name=('loc_loss_cls_%d_box_%d' % (class_idx, bi)))
        t_classwise_loc_loss_update_ops[
            ('loc_loss_cls_%d_box_%d' % (class_idx, bi))] = t_mean_loss_op
    classwise_loc_loss = t_classwise_loc_loss[class_idx]
    t_mean_loss, t_mean_loss_op = tf.compat.v1.metrics.mean(
        classwise_loc_loss,
        name=('loc_loss_cls_%d' % class_idx))
    t_classwise_loc_loss_update_ops[
        ('loc_loss_cls_%d' % class_idx)] = t_mean_loss_op

# metrics
t_recall_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_recall, t_recall_update_op = tf.compat.v1.metrics.recall(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('recall_%d' % class_idx))
    t_recall_update_ops[('recall_%d' % class_idx)] = t_recall_update_op

t_precision_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_precision, t_precision_update_op = tf.compat.v1.metrics.precision(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        tf.equal(t_predictions, tf.constant(class_idx, tf.int32)),
        name=('precision_%d' % class_idx))
    t_precision_update_ops[('precision_%d' % class_idx)] = t_precision_update_op

t_mAP_update_ops = {}
for class_idx in range(NUM_CLASSES):
    t_mAP, t_mAP_update_op = tf.compat.v1.metrics.auc(
        tf.equal(t_class_labels, tf.constant(class_idx, tf.int32)),
        t_probs[:, class_idx],
        num_thresholds=200,
        curve='PR',
        name=('mAP_%d' % class_idx),
        summation_method='careful_interpolation')
    t_mAP_update_ops[('mAP_%d' % class_idx)] = t_mAP_update_op

t_mean_cls_loss, t_mean_cls_loss_op = tf.compat.v1.metrics.mean(
    t_cls_loss,
    name='mean_cls_loss')
t_mean_loc_loss, t_mean_loc_loss_op = tf.compat.v1.metrics.mean(
    t_loc_loss,
    name='mean_loc_loss')
t_mean_reg_loss, t_mean_reg_loss_op = tf.compat.v1.metrics.mean(
    t_reg_loss,
    name='mean_reg_loss')
t_mean_total_loss, t_mean_total_loss_op = tf.compat.v1.metrics.mean(
    t_total_loss,
    name='mean_total_loss')

metrics_update_ops = {
    'cls_loss': t_mean_cls_loss_op,
    'loc_loss': t_mean_loc_loss_op,
    'reg_loss': t_mean_reg_loss_op,
    'total_loss': t_mean_total_loss_op, }
metrics_update_ops.update(t_recall_update_ops)
metrics_update_ops.update(t_precision_update_ops)
metrics_update_ops.update(t_mAP_update_ops)
metrics_update_ops.update(t_classwise_loc_loss_update_ops)

# optimizers ================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'pred_box': t_pred_box

}
fetches.update(metrics_update_ops)


# preprocessing data ========================================================
class DataProvider(object):
    """This class provides input data to training.
    It has option to load dataset in memory so that preprocessing does not
    repeat every time.
    Note, if there is randomness inside graph creation, samples should be
    reloaded for the randomness to take effect.
    """

    def __init__(self, fetch_data, load_dataset_to_mem=True,
                 load_dataset_every_N_time=1, capacity=1):
        self._fetch_data = fetch_data
        self._loaded_data_dic = {}
        self._loaded_data_ctr_dic = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity

    def provide(self, frame_idx):
        extend_frame_idx = frame_idx + np.random.choice(
            self._capacity) * NUM_TEST_SAMPLE
        if self._load_dataset_to_mem:
            if extend_frame_idx in self._loaded_data_ctr_dic:
                ctr = self._loaded_data_ctr_dic[extend_frame_idx]
                if ctr >= self._load_every_N_time:
                    del self._loaded_data_ctr_dic[extend_frame_idx]
                    del self._loaded_data_dic[extend_frame_idx]
            if frame_idx not in self._loaded_data_dic:
                self._loaded_data_dic[extend_frame_idx] = self._fetch_data(
                    frame_idx)
                self._loaded_data_ctr_dic[extend_frame_idx] = 0
            self._loaded_data_ctr_dic[extend_frame_idx] += 1
            return self._loaded_data_dic[extend_frame_idx]
        else:
            return self._fetch_data(frame_idx)


data_provider = DataProvider(fetch_data, load_dataset_to_mem=False)
saver = tf.compat.v1.train.Saver()
graph = tf.compat.v1.get_default_graph()
if meta_config['gpu_memusage'] < 0:
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
else:
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=meta_config['gpu_memusage'])

statistics_path = os.path.join(meta_config['train_dir'], f'{split}.json')
if not os.path.exists(statistics_path):
    initialize_stats_dict(NUM_CLASSES, statistics_path)

def eval_once(graph, gpu_options, saver, checkpoint_path):
    """Evaluate the model once. """
    with tf.compat.v1.Session(graph=graph,
                              config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.global_variables()))
        sess.run(tf.compat.v1.variables_initializer(tf.compat.v1.local_variables()))
        print('Restore from checkpoint %s' % checkpoint_path)
        saver.restore(sess, checkpoint_path)
        previous_step = sess.run(global_step)
        print('Global step = %d' % previous_step)
        start_time = time.time()
        for frame_idx in range(NUM_TEST_SAMPLE):
            (input_v, vertex_coord_list, keypoint_indices_list, edges_list,
             cls_labels, encoded_boxes, valid_boxes) \
                = data_provider.provide(frame_idx)
            feed_dict = {
                t_initial_vertex_features: input_v,
                t_class_labels: cls_labels,
                t_encoded_gt_boxes: encoded_boxes,
                t_valid_gt_boxes: valid_boxes,
                t_is_training: config['eval_is_training'],
            }
            feed_dict.update(dict(zip(t_edges_list, edges_list)))
            feed_dict.update(
                dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
            feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
            results = sess.run(fetches, feed_dict=feed_dict)

            if NUM_TEST_SAMPLE >= 10:
                if (frame_idx + 1) % (NUM_TEST_SAMPLE // 10) == 0:
                    print('@frame %d' % frame_idx)
                    print('cls:%f, loc:%f, reg:%f, loss: %f'
                          % (results['cls_loss'], results['loc_loss'],
                             results['reg_loss'], results['total_loss']))
                    for class_idx in range(NUM_CLASSES):
                        print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                              % (class_idx,
                                 results['recall_%d' % class_idx],
                                 results['precision_%d' % class_idx],
                                 results['mAP_%d' % class_idx],
                                 results['loc_loss_cls_%d' % class_idx]))
                        print('         ' + \
                              'x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f'
                              % (
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 0)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 1)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 2)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 3)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 4)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 5)],
                                  results['loc_loss_cls_%d_box_%d' % (class_idx, 6)]),
                              )
        epoch_time = time.time() - start_time
        print('STEP: %d, time cost: %f'
              % (results['step'], epoch_time))
        print('cls:%f, loc:%f, reg:%f, loss: %f'
              % (results['cls_loss'], results['loc_loss'], results['reg_loss'],
                 results['total_loss']))
        for class_idx in range(NUM_CLASSES):
            print('Class_%d: recall=%f, prec=%f, mAP=%f, loc=%f'
                  % (class_idx,
                     results['recall_%d' % class_idx],
                     results['precision_%d' % class_idx],
                     results['mAP_%d' % class_idx],
                     results['loc_loss_cls_%d' % class_idx]))
            print("         x=%.4f y=%.4f z=%.4f l=%.4f h=%.4f w=%.4f y=%.4f"
                  % (
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 0)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 1)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 2)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 3)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 4)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 5)],
                      results['loc_loss_cls_%d_box_%d' % (class_idx, 6)]),
                  )
        # add summaries ====================================================
        """for key in metrics_update_ops:
            write_summary_scale(key, results[key], results['step'],
                                meta_config['eval_dir'])"""
        # save epoch statistics to file in checkpoint
        append_epoch_to_stats_dict(results, NUM_CLASSES, statistics_path, epoch_time, mode='eval')

        return results['step'].item(), results['total_loss'].item(), results['mAP_1'].item()

def extract_checkpoint_list_from_tf_checkpoint_file(checkpoint_path):
    with open(checkpoint_path) as ckpt:
        lines = ckpt.readlines()
        relevant_ckpt_lines = lines[2:]
        relevant_ckpts = []
        for ckpt_line in relevant_ckpt_lines:
            ckpt_line = ckpt_line.replace('all_model_checkpoint_paths: "model-', '').replace('"\n', '')
            relevant_ckpts.append(int(ckpt_line))
    return relevant_ckpts


def eval_repeat(checkpoint_path):

    ckpts_to_evaluate = extract_checkpoint_list_from_tf_checkpoint_file(os.path.join(checkpoint_path, 'checkpoint'))
    for ckpt in ckpts_to_evaluate:
        tf.compat.v1.reset_default_graph()
        current_model = f'model-{ckpt}'
        model_path = os.path.join(checkpoint_path, current_model)
        eval_once(graph, gpu_options, saver, model_path)

def eval_while_training():
    early_stopping = 'early_stopping' in meta_config and meta_config['early_stopping'] is True
    current_step = 0
    if early_stopping:
        #smallest_loss = float('inf')
        highest_mAP = -float('inf')
        epochs_since_best = 0
        patience = meta_config['decay_patience']
        best_path = f'./checkpoints/{checkpoint}/best'

        history_path = f'{checkpoint_path}/history'
        if os.path.exists(history_path):
            history = load_train_config(history_path)
        else:
            history = {'eval_on': DATASET_SPLIT_FILE}
        lr = meta_config['initial_lr']
    last_evaluated_model_path = None
    no_new_ckpt_since = 0
    while True:
        previous_time = time.time()
        model_path = tf.train.latest_checkpoint(meta_config['train_dir'])
        if not model_path or last_evaluated_model_path == model_path or '1400000' in model_path or '1000000' in model_path:
            print('No current checkpoint to evaluate found in %s, wait for %f seconds'
                  % (meta_config['train_dir'], meta_config['eval_every_second']))
            no_new_ckpt_since += meta_config['eval_every_second']
            if no_new_ckpt_since >= meta_config['eval_every_second']*128:
                break
        else:
            no_new_ckpt_since = 0
            last_evaluated_model_path = model_path
            current_step, val_loss, mAP = eval_once(graph, gpu_options, saver, model_path)
            if split != 'test':
                evaluate_json_statistics(checkpoint, NUM_CLASSES, mode=f'{split}', show_plot=False,
                                         save_frequency=meta_config['save_every_epoch'])

            current_epoch = current_step/meta_config['NUM_TEST_SAMPLE']
            if current_epoch >= meta_config['max_epoch'] or split == 'test':
                break

            if early_stopping:
                """if val_loss < smallest_loss:
                    smallest_loss = val_loss
                    epochs_since_best = 0
                    # beste immer sichern (sonst overfitting seit patience Epochen)
                    meta_config['best_epoch'] = int(current_epoch)
                    save_config(meta_config_path, meta_config)"""
                if mAP > highest_mAP:
                    highest_mAP = mAP
                    epochs_since_best = 0
                    history[f'lr{lr}'] = {}
                    history[f'lr{lr}']['highest_mAP'] = highest_mAP
                    history[f'lr{lr}']['best_epoch'] = current_epoch
                    history[f'lr{lr}']['best_model'] = current_step
                    save_config(history_path, history)

                    if os.path.exists(best_path):
                        shutil.rmtree(best_path)
                    os.mkdir(best_path)
                    shutil.copyfile(f'./checkpoints/{checkpoint}/model-{current_step - 1}.data-00000-of-00001',
                                    f'{best_path}/model-{current_step-1}.data-00000-of-00001')
                    shutil.copyfile(f'./checkpoints/{checkpoint}/model-{current_step-1}.index',
                                    f'{best_path}/model-{current_step-1}.index')
                    shutil.copyfile(f'./checkpoints/{checkpoint}/model-{current_step-1}.meta',
                                    f'{best_path}/model-{current_step-1}.meta')
                elif epochs_since_best >= patience:
                    meta_config['max_epoch'] = int(current_epoch)
                    save_config(meta_config_path, meta_config)
                else:
                    epochs_since_best += 1

        time_to_next_eval = (previous_time + meta_config['eval_every_second'] - time.time())
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)


if __name__ == '__main__':
    if eval_while_training_on:
        eval_while_training()
    else:
        eval_repeat(checkpoint_path)
    if split != 'test':
        evaluate_json_statistics(checkpoint, NUM_CLASSES, mode=f'{split}', show_plot=False,
                                 save_frequency=meta_config['save_every_epoch'])

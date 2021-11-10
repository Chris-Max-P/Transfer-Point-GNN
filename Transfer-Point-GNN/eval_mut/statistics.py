from math import sqrt

import matplotlib.pyplot as plt
from source.dataset_classes.data_specifics import get_label_map
import os
import json

def initialize_stats_dict(num_classes, statistics_path):
    epochs_dict = {
        'step_list': [],
        'lr_list': [],
        'time_cost_list': [],
        'cls_loss_list': [],
        'loc_loss_list': [],
        'reg_loss_list': [],
        'total_loss_list': [],
    }  # epoch_id = index + max_epoch from previous learning

    for class_idx in range(num_classes):
        class_dict = {
            'recall': [],
            'precision': [],
            'mAP': [],
            'loc_loss_cls': [],
            'x': [],
            'y': [],
            'z': [],
            'l': [],
            'h': [],
            'w': [],
            'yaw': []
        }
        epochs_dict[class_idx] = class_dict

    with open(statistics_path, 'w') as stats:
        stats.write(json.dumps(epochs_dict, indent=4))

def append_epoch_to_stats_dict(results, num_classes, statistics_path, epoch_time, mode='train'):
    with open(statistics_path, 'r') as stats:
        epochs_dict = json.loads(stats.read())

        current_step = results['step'].item()

        if current_step in epochs_dict['step_list']:
            index = epochs_dict['step_list'].index(current_step)
            epochs_dict['step_list'][index] = current_step
            if mode == 'train':
                epochs_dict['lr_list'][index] = (results['learning_rate'].item())
            epochs_dict['time_cost_list'][index] = epoch_time
            epochs_dict['cls_loss_list'][index] = (results['cls_loss'].item())
            epochs_dict['loc_loss_list'][index] = (results['loc_loss'].item())
            epochs_dict['reg_loss_list'][index] = (results['reg_loss'].item())
            epochs_dict['total_loss_list'][index] = (results['total_loss'].item())

            for class_idx in range(num_classes):
                class_dict = epochs_dict[f'{class_idx}']
                class_dict['recall'][index] = (results['recall_%d' % class_idx].item())
                class_dict['precision'][index] = (results['precision_%d' % class_idx].item())
                class_dict['mAP'][index] = (results['mAP_%d' % class_idx].item())
                class_dict['loc_loss_cls'][index] = (results['loc_loss_cls_%d' % class_idx].item())
                class_dict['x'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 0)].item())
                class_dict['y'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 1)].item())
                class_dict['z'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 2)].item())
                class_dict['l'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 3)].item())
                class_dict['h'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 4)].item())
                class_dict['w'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 5)].item())
                class_dict['yaw'][index] = (results['loc_loss_cls_%d_box_%d' % (class_idx, 6)].item())
        else:
            epochs_dict['step_list'].append(current_step)
            if mode == 'train':
                epochs_dict['lr_list'].append(results['learning_rate'].item())
            epochs_dict['time_cost_list'].append(epoch_time)
            epochs_dict['cls_loss_list'].append(results['cls_loss'].item())
            epochs_dict['loc_loss_list'].append(results['loc_loss'].item())
            epochs_dict['reg_loss_list'].append(results['reg_loss'].item())
            epochs_dict['total_loss_list'].append(results['total_loss'].item())

            for class_idx in range(num_classes):
                class_dict = epochs_dict[f'{class_idx}']
                class_dict['recall'].append(results['recall_%d' % class_idx].item())
                class_dict['precision'].append(results['precision_%d' % class_idx].item())
                class_dict['mAP'].append(results['mAP_%d' % class_idx].item())
                class_dict['loc_loss_cls'].append(results['loc_loss_cls_%d' % class_idx].item())
                class_dict['x'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 0)].item())
                class_dict['y'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 1)].item())
                class_dict['z'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 2)].item())
                class_dict['l'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 3)].item())
                class_dict['h'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 4)].item())
                class_dict['w'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 5)].item())
                class_dict['yaw'].append(results['loc_loss_cls_%d_box_%d' % (class_idx, 6)].item())

    with open(statistics_path, 'w') as stats:
        stats.write(json.dumps(epochs_dict, indent=4))

def evaluate_json_statistics(checkpoint, num_classes, mode='train', show_plot=True, save_frequency=1):
    checkpoint_path = f'./checkpoints/{checkpoint}'
    class_dict = {num:obj for obj,num in get_label_map(checkpoint).items()}
    with open(os.path.join(checkpoint_path, f'{mode}.json'), 'r') as stats:
        epochs_dict = json.loads(stats.read())
    num_plots = int(num_classes/2)

    fig, plots = plt.subplots(num_plots)

    # loss plot
    total_loss_list = epochs_dict['total_loss_list']
    epoch_list = [i*save_frequency for i in range(len(total_loss_list))]

    cls_loss_list = epochs_dict['cls_loss_list']
    loc_loss_list = epochs_dict['loc_loss_list']
    #reg_loss_list = epochs_dict['reg_loss_list']

    plots[0].plot(epoch_list[1:], total_loss_list[1:], 'tab:orange', label='Loss')
    #plots[0].plot(epoch_list[1:], cls_loss_list[1:], label='cls_loss')
    #plots[0].plot(epoch_list[1:], loc_loss_list[1:], label='loc_loss')
    #plots[0].plot(epoch_list[1:], reg_loss_list[1:], label='reg_loss')

    plots[0].set_title('General')
    #plots[0].set_xlabel('epochs')
    plots[0].set_ylabel('')
    plots[0].legend()

    # class metrics plots
    for i in range(1, num_classes-2, 2):
        cls_plot = plots[i//2 + 1]
        recall_list = epochs_dict[f'{i}']['recall']
        #precision_list = epochs_dict[f'{i}']['precision']
        mAP_list = epochs_dict[f'{i}']['mAP']
        #loc_loss_cls_list = epochs_dict[f'{i}']['loc_loss_cls']

        #cls_plot.plot(epoch_list, recall_list, label='Recall')
        #cls_plot.plot(epoch_list, precision_list, label='Precision')
        cls_plot.plot(epoch_list, mAP_list, 'tab:green', label='mAP')
        #cls_plot.plot(epoch_list, loc_loss_cls_list, label='Loc loss cls')

        cls_plot.set_title(class_dict[i])
        cls_plot.set_xlabel('epochs')
        cls_plot.set_ylabel('mAP')
        cls_plot.legend()

    plt.savefig(os.path.join(checkpoint_path, f'{mode}_{checkpoint}.png'))
    if show_plot:
        plt.show()

def compare_lr(num_trained_layers, lr_list, mode='eval', show_plot=True):
    save_frequency = 1
    if mode == 'eval':
        save_frequency = 5
    fig, plots = plt.subplots(2)
    for lr in lr_list:
        checkpoint_path = f'./checkpoints/posts_rot_1024_{num_trained_layers}_lr{lr}'
        if not os.path.exists(checkpoint_path):
            return
        with open(os.path.join(checkpoint_path, f'{mode}.json'), 'r') as stats:
            epochs_dict = json.loads(stats.read())

            mAP_list = epochs_dict['1']['mAP']
            epoch_list = [i*save_frequency for i in range(len(mAP_list))]
            plots[0].plot(epoch_list, mAP_list, label=f'lr {lr}')

            total_loss_list = epochs_dict['total_loss_list']
            plots[1].plot(epoch_list[2:], total_loss_list[2:], label=f'LR {lr}')

    values = ['mAP', 'loss']
    plots[0].set_title(f'LR compared for {num_trained_layers} trained layers')
    plots[1].set_xlabel('epochs')
    for index, value in enumerate(values):
        plots[index].set_ylabel(value)
        plots[index].legend()
    if show_plot:
        plt.show()
    fig.savefig(f'./checkpoints/compare_{mode}_of_{num_trained_layers}_lr.png')

def compare_num_layers():
    fig, plots = plt.subplots()
    num_layers = [1,5]
    for num in num_layers:
        checkpoint_path = f'./checkpoints/posts_rot_1024_{num}'
        if not os.path.exists(checkpoint_path):
            print('Checkpoint path does not exist.')
            return
        with open(os.path.join(checkpoint_path, 'meta_config')) as meta_config_file:
            meta_config = json.loads(meta_config_file.read())
            max_epoch = meta_config['max_epoch']
            with open(os.path.join(checkpoint_path, 'eval.json'), 'r') as stats:
                epochs_dict = json.loads(stats.read())

                mAP_list = epochs_dict['1']['mAP'][:max_epoch]
                epoch_list = [i for i in range(len(mAP_list))]
                plots.plot(epoch_list, mAP_list, label=f'{num} layer')

    plots.set_title(f'Training from scratch (5 layer) vs. fine-tuning 1 layer')
    plots.set_xlabel('epochs')
    plots.set_ylabel('mAP')
    plots.legend()

    fig.savefig(f'./checkpoints/compare_layers_1_vs_5.png')
    pass

def compare_num_labels(num_trained_layers):
    num_labels = [1024, 786, 512, 256, 128]
    fig, plots = plt.subplots()
    for num in num_labels:
        checkpoint_path = f'./checkpoints/posts_rot_{num}_{num_trained_layers}'
        if not os.path.exists(checkpoint_path):
            return
        with open(os.path.join(checkpoint_path, 'meta_config')) as meta_config_file:
            meta_config = json.loads(meta_config_file.read())
            max_epoch = meta_config['max_epoch']
            with open(os.path.join(checkpoint_path, 'val.json'), 'r') as stats:
                epochs_dict = json.loads(stats.read())

                mAP_list = epochs_dict['1']['mAP'][:max_epoch]
                epoch_list = [i for i in range(len(mAP_list))]
                plots.plot(epoch_list, mAP_list, label=f'{num} labels')

            """total_loss_list = epochs_dict['total_loss_list'][:max_epoch]
            plots[1].plot(epoch_list[2:], total_loss_list[2:], label=f'{num} labels')"""

    plots.set_title(f'Amount of used labels compared for {num_trained_layers} trained layers')
    plots.set_xlabel('epochs')
    plots.legend()
    plots.set_ylabel('mAP')

    fig.savefig(f'./checkpoints/compare_labels_of_{num_trained_layers}.png')


def variance(stats_list):
    mittelwert = sum(stats_list) / len(stats_list)
    temp = []
    for stat in stats_list:
        temp.append((mittelwert - stat)**2)
    variance = sum(temp)/len(stats_list)
    return mittelwert, variance, sqrt(variance)

"""def get_test_result_data(NUM_CLASSES):
    results = {
        'step': 1,
        'lr': 0.125,
        'time_cost': 'time',
        'cls_loss': 'cls_loss',
        'loc_loss': 'loc_loss',
        'reg_loss': 'reg_loss',
        'total_loss': 'total_loss'
    }

    for class_idx in range(NUM_CLASSES):
        results['recall_%d' % class_idx] = f'recall_{class_idx}'
        results['precision_%d' % class_idx] = f'precision_{class_idx}'
        results['mAP_%d' % class_idx] = f'mAP_{class_idx}'
        results['loc_loss_cls_%d' % class_idx] = f'loc_loss_cls_{class_idx}'

        results['loc_loss_cls_%d_box_%d' % (class_idx, 0)] = f'loc_loss_cls_{class_idx}_box_{0}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 1)] = f'loc_loss_cls_{class_idx}_box_{1}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 2)] = f'loc_loss_cls_{class_idx}_box_{2}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 3)] = f'loc_loss_cls_{class_idx}_box_{3}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 4)] = f'loc_loss_cls_{class_idx}_box_{4}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 5)] = f'loc_loss_cls_{class_idx}_box_{5}'
        results['loc_loss_cls_%d_box_%d' % (class_idx, 6)] = f'loc_loss_cls_{class_idx}_box_{6}'
    return results"""



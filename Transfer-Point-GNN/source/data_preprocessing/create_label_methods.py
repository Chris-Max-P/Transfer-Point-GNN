from shutil import copyfile
from util_point_gnn.config_util import *
from source.data_preprocessing.label_file_operations import *
from source.dataset_classes.data_specifics import *


def create_config(label_method, num_classes,
                  trainable_layers=[False, False, False, False, True]):
    if num_classes == 4:
        config = load_config('../../configs/default_finetune_config')
    if num_classes == 6:
        config = load_config('../../configs/posts_rot_128_1_config')
    config['label_method'] = label_method
    for index, layer_config in enumerate(config['model_kwargs']['layer_configs']):
        layer_config['trainable'] = trainable_layers[index]
    config['num_classes'] = num_classes
    finetuned_layers = trainable_layers.count(True)
    save_config(f'../configs/{label_method}_{finetuned_layers}_config', config)

def create_meta_config(label_method, num_files, finetuned_layers=1):
    meta_config = load_config('../../configs/default_finetune_meta_config')
    meta_config['NUM_TEST_SAMPLE'] = num_files
    meta_config['train_dataset'] = f'{label_method}_train.txt'
    meta_config['val_dataset'] = f'{label_method}_val.txt'
    meta_config['train_dir'] = f'checkpoints/{label_method}_{finetuned_layers}'
    max_epoch = meta_config['max_epoch']
    max_steps = num_files * max_epoch
    meta_config['max_steps'] = max_steps
    meta_config['decay_step'] = max_steps//3
    save_config(f'../configs/{label_method}_{finetuned_layers}_meta_config', meta_config)

def create_checkpoint(label_method, finetuned_layers=1, classes=6):
    os.makedirs(f'../checkpoints/{label_method}_{finetuned_layers}', exist_ok=True)
    if finetuned_layers > 0:
        if classes == 4:
            source = '../checkpoints/car_auto_T3_train'
            target = f'../checkpoints/{label_method}_{finetuned_layers}'
            copyfile(f'{source}/checkpoint', f'{target}/checkpoint')
            copyfile(f'{source}/model-1400000.data-00000-of-00001', f'{target}/model-1400000.data-00000-of-00001')
            copyfile(f'{source}/model-1400000.index', f'{target}/model-1400000.index')
            copyfile(f'{source}/model-1400000.meta', f'{target}/model-1400000.meta')
        if classes == 6:
            source = '../checkpoints/ped_cyl_auto_T3_trainval'
            target = f'../checkpoints/{label_method}_{finetuned_layers}'
            copyfile(f'{source}/checkpoint', f'{target}/checkpoint')
            copyfile(f'{source}/model-1000000.data-00000-of-00001', f'{target}/model-1000000.data-00000-of-00001')
            copyfile(f'{source}/model-1000000.index', f'{target}/model-1000000.index')
            copyfile(f'{source}/model-1000000.meta', f'{target}/model-1000000.meta')


def create_label_method(label_methods_list, make_splits=True, divide_splits=True):
    """
    Creates splits, configs and checkpoints for given label_methods.
    Label_methods have to be inserted into data_specifics.py first.
    """
    if make_splits:
        make_split_files_for_configurations(label_methods_list)
    if divide_splits:
        divide_splits(split_dir, label_methods_list)
    splits_counted_list, obj_count_dict_list, num_files_list = get_MUT_split_object_counts(label_methods_list)

    for index, split_file in enumerate(splits_counted_list):
        if 'train' in split_file:
            label_method = split_file.replace('_train','').replace('_val','').replace('_test','').split('.')[0]
            num_classes = get_label_map(label_method).get('DontCare') + 1
            num_files = num_files_list[index]

            create_config(label_method, num_classes)
            create_meta_config(label_method, num_files)
            create_checkpoint(label_method, classes=num_classes)
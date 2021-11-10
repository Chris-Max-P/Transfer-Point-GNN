import shutil
from source.dataset_classes.data_specifics import label_map_to_simple_list, get_label_map
from util.paths_and_data import *
import os

label_order = ['name', 'truncation', 'occlusion', 'alpha',
               'xmin', 'ymin', 'xmax', 'ymax',
               'height', 'width', 'length',
               'x3d', 'y3d', 'z3d', 'yaw', 'score', 'eval_result']

def replace_text_for_all_label_files(old, new, labels_dir):
    for txt_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, txt_file), 'r', encoding="utf-8", errors="ignore") as file:
            content = file.read()
            new_content = content.replace(old, new)  # .replace('"', "")
        with open(os.path.join(labels_dir, txt_file), 'w', encoding="utf-8", errors="ignore") as file:
            file.write(new_content)


def divide_splits(splits_dir, split_file_name_list):
    """
    Divides a split file (containing all label file names for a specific label_method)
    in a train, a val and a test file.
    :param splits_dir: the directory in which split files are located
    :param split_file_name: the name of the split file to be splitted
    """
    for split_file_name in split_file_name_list:
        with open(os.path.join(splits_dir, split_file_name + '.txt')) as split_file:
            train_split = f'{split_file_name}_train.txt'
            val_split = f'{split_file_name}_val.txt'
            test_split = f'{split_file_name}_test.txt'
            lines = split_file.readlines()
            with open(os.path.join(splits_dir, train_split), 'a') as train:
                with open(os.path.join(splits_dir, val_split), 'a') as val:
                    with open(os.path.join(splits_dir, test_split), 'a') as test:
                        for index, line in enumerate(lines, 1):
                            mod = index % 5
                            if mod == 0 or mod == 1 or mod == 2:
                                file_to_write = train
                            if mod == 3:
                                file_to_write = val
                            if mod == 4:
                                file_to_write = test
                            file_to_write.write(line)


def remove_points_from_file_names(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        os.rename(full_path, full_path.replace(".00", ""))


def unpack_directories(from_path, to_path):
    for file in os.listdir(from_path):
        full_path = os.path.join(from_path, file)
        dest_path = os.path.join(to_path, file)
        shutil.move(full_path, dest_path)


def readFile(path, enc=None):
    with open(path, 'r', encoding=enc) as file:
        lines = file.read()
    return lines


def change_encoding(file_path):
    """
    Changes encoding of an utf-16 file to utf-8,
    exchanging some relevant special characters.
    """
    content = readFile(file_path, 'utf-16')
    content = content.replace('ä', 'ae')
    content = content.replace('ö', 'oe')
    content = content.replace('ü', 'ue')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def change_enc_for_all_files(path):
    for file in os.listdir(path):
        change_encoding(os.path.join(path, file))


def change_order_of_label_info(label_dir):
    """
    Changes the order of the specified label info fields.
    Has to be adapted to the special case in which it is needed.
    """
    files = os.listdir(label_dir)
    for file in files:
        label_file = os.path.join(label_dir, file)
        new_content = ''
        with open(label_file, 'r') as labels:
            lines = labels.readlines()
            for line in lines:
                if line == '':
                    continue
                fields = line.split(' ')
                # x,y,z -> y,z,x
                x = fields[11]
                y = fields[12]
                z = fields[13]

                fields[11] = y
                fields[12] = z
                fields[13] = x

                # l,h,w -> h,w,l
                l = fields[8]
                h = fields[9]
                w = fields[10]

                fields[8] = h
                fields[9] = w
                fields[10] = l

                new_line = ''
                for field in fields:
                    if field != '0\n':
                        new_line += field + ' '
                    else:
                        new_line += field
                new_content += new_line

        with open(label_file, 'w') as labels:
            labels.write(new_content)

def label_list_to_label_line(label_fields_list):
    label_line = ''
    for field in label_fields_list:
        if '\n' not in field:
            label_line += field + ' '
        else:
            label_line += field
    return label_line

def label_dict_to_label_line(label_dict):
    label_line = ''
    entries = label_order
    if 'eval_result' not in label_dict:
        entries = entries[:-1]
    for entry in entries:
        if entry in label_dict:
            label_line += str(label_dict[entry]) + ' '
    label_line += '\n'
    return label_line

def change_label_info(label_dir, object, position, new_info):
    """
    Changes the info for every label
    for the given objects at the given position to new_info
    positions:
    0 -> name
    1 -> truncation
    2 -> occlusion
    3 -> alpha
    4 -> xmin
    5 -> ymin
    6 -> xmax
    7 -> ymax
    8 -> height
    9 -> width
    10 -> length
    11 -> x3d
    12 -> y3d
    13 -> z3d
    14 -> yaw
    15 -> score
    """
    files = os.listdir(label_dir)
    for file in files:
        label_file = os.path.join(label_dir, file)
        new_content = ''
        with open(label_file, 'r') as labels:
            lines = labels.readlines()
            for line in lines:
                if line == '':
                    continue
                field_list = line.split(' ')
                if field_list[0] != object:
                    new_content += line
                    continue
                field_list[position] = new_info
                new_content += label_list_to_label_line(field_list)

        with open(label_file, 'w') as labels:
            labels.write(new_content)

def remove_bad_filenames_from_splits():
    import platform
    platform = platform.system()

    if platform == 'Windows':
        bad_files_txt = 'F:\\TransferPoint-GNN\\MUT\\bad_files.txt'
        mut_splits = "F:\\TransferPoint-GNN\\MUT\\3DOP_splits"
    else:
        bad_files_txt = '/home/stud/paul/dataset/MUT/bad_files.txt'
        mut_splits = "/home/stud/paul/dataset/MUT/3DOP_splits"

    with open(bad_files_txt) as bads:
        bad_files = bads.read().replace('.e57', '.txt')

    splits = os.listdir(mut_splits)

    for split in splits:
        lines = []
        with open(os.path.join(mut_splits, split)) as txt_file:
            lines = txt_file.readlines()

        with open(os.path.join(mut_splits, split), 'w') as txt_file:
            removed_lines = []
            for line in lines:
                if line not in bad_files:
                    txt_file.write(line)
                else:
                    removed_lines.append(line)
            print(f'Removed: {removed_lines}')


"""
def add_missing_label_files(tunnel_dir, label_dir):
    tunnels = os.listdir(tunnel_dir)
    label = os.listdir(label_dir)
    to_create = []

    for e57_name in tunnels:
        e57_name = e57_name.replace('.e57', '.txt')
        if e57_name not in label:
            to_create.append(e57_name)

    for name in to_create:
        with open(os.path.join(label_dir, name), mode='w') as new_label_file:
            new_label_file.write('')
"""

def make_Stab_rot_sym(label_dir):
    change_label_info(label_dir,'SR-Stab(IST-Lage)',9,'0.1')
    replace_text_for_all_label_files("(IST-Lage)", "", label_dir)
    replace_text_for_all_label_files("(SOLL-Lage)", "", label_dir)


def string_contains_obj_from_list(string, list):
    for obj in list:
        if obj in string:
            return True
    return False

def filter_label_files(label_dir, obj_list):
    """
    Filters the given label dir for all files,
    that contain any of the given objects.
    :param label_dir: where to search
    :param obj_list: what to search for
    :return: list of files containing given objects
    """
    label_files = os.listdir(label_dir)
    obj_containing_files = []
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path) as current_label_file:
            content = current_label_file.read()
            if string_contains_obj_from_list(content, obj_list):
                obj_containing_files.append(label_file)
    return obj_containing_files

def get_label_file_list_containing_certain_number_of_objects(
        label_dir, obj, max_num):
    label_files = os.listdir(label_dir)
    obj_containing_files = []
    rest_label_files = []
    obj_count = 0
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path) as current_label_file:
            content = current_label_file.read()
            current_file_count = content.count(obj)
            obj_count += current_file_count
            if current_file_count > 0:
                if obj_count + current_file_count <= max_num:
                    obj_containing_files.append(label_file)
                else:
                    rest_label_files.append(label_file)
    return obj_containing_files, rest_label_files

def write_label_files_to_split(target_split_file, file_list):
    """
    Writes the given label files to the target split file.
    :param target_split_file: where to write
    :param file_list: what to write
    """
    content = ''
    for file in file_list:
        content += file + '\n'
    with open(target_split_file, mode='w') as target:
        target.write(content)

def filter_labels_and_save_to_split(label_dir, target_split_file, obj_list, count=None):
    """
    Creates a split file containing the desired objects. If a number for an object is given
    (obj_list must have size 1 in this case), creates 2 split files:
        - one containing the given number of objects for training
        - one containing all other known objects from this type
    """
    if count is None:
        files_to_save = filter_label_files(label_dir, obj_list)
        write_label_files_to_split(target_split_file, files_to_save)
    elif count is not None:
        files_to_save, rest_label_files = get_label_file_list_containing_certain_number_of_objects(
            label_dir, obj_list[0], count)
        write_label_files_to_split(
            target_split_file[:-4] + '_train' + target_split_file[-4:], files_to_save)
        write_label_files_to_split(
            target_split_file[:-4] + '_rest' + target_split_file[-4:], rest_label_files)

def make_split_files_for_configurations(label_methods_list):
    """
    Creates split files for all label_methods given
    """
    target_split_file_list = []
    for label_method in label_methods_list:
        target_split_file_list.append(os.path.join(split_dir, label_method + '.txt'))
    assert(len(label_methods_list) == len(target_split_file_list))
    for label_method, target_split_file in zip(label_methods_list, target_split_file_list):
        obj_list = label_map_to_simple_list(get_label_map(label_method))
        count = None
        try:
            max_num = int(label_method.split('_')[-1])
            if max_num > 0:
                count = max_num
        except ValueError:
            pass
        filter_labels_and_save_to_split(label_dir, target_split_file, obj_list, count=count)


def count_objects_in_split(obj_list, split_file_path, label_dir, KITTI=False):
    appendix = ''
    if KITTI:
        appendix += '.txt'
    obj_count_dict = {}
    for obj in obj_list:
        obj_count_dict[obj] = 0
    with open(split_file_path) as split_file:
        label_files = split_file.readlines()
    for label_file in label_files:
        label_file = label_file.replace('\n', '')
        with open(os.path.join(label_dir, label_file + appendix)) as current_file:
            labels = current_file.readlines()
        for label in labels:
            obj_name = label.split(' ')[0]
            try:
                obj_count_dict[obj_name] += 1
            except NameError and KeyError:
                continue
    return obj_count_dict, len(label_files)


def get_MUT_split_object_counts(label_method_list=None, output=True):
    """
    Counts all objects either in all MUT splits or in just the given ones.
    :returns a triple of
        - a list of splits that were counted
        - a list of obj. counts
        - a list of number of files in each split
    """
    from source.dataset_classes.dataset import get_label_map

    splits_counted_list = []
    obj_count_dict_list = []
    num_files_list = []
    for split_file in os.listdir(split_dir):
        skip = True
        if label_method_list is None or any(label_method in split_file for label_method in label_method_list):
            skip = False
        if label_method_list is not None and skip:
            continue
        # get label_method for corresponding object list
        label_method = split_file.replace('_train','').replace('_val','').replace('_test','').split('.')[0]
        obj_list = get_label_map(label_method)

        split_file_path = os.path.join(split_dir, split_file)
        if 'rot' in split_file_path:
            labels_dir = rot_sym_label_dir
        else:
            labels_dir = label_dir
        obj_count_dict, num_files = count_objects_in_split(obj_list, split_file_path, labels_dir)

        splits_counted_list.append(split_file)
        obj_count_dict_list.append(obj_count_dict)
        num_files_list.append(num_files)

        if output:
            print(split_file)
            print(obj_count_dict)
            print(num_files)
            print('')
    return splits_counted_list, obj_count_dict_list, num_files_list

def count_objects_in_KITTI_split():
    obj_list = label_map_to_simple_list(get_label_map('yaw'))
    kitti_split = 'F:\\TransferPoint-GNN\\KITTI\\3DOP_splits\\train.txt'
    kitti_label_dir = 'F:\\TransferPoint-GNN\\KITTI\\labels\\training\\label_2'
    return count_objects_in_split(obj_list, kitti_split, kitti_label_dir, KITTI=True)


def count_objects_in_MUT_labels(label_dir):
    obj_list = label_map_to_simple_list(get_label_map('all'))

    obj_count_dict = {}
    for obj in obj_list:
        obj_count_dict[obj] = 0

    label_files = os.listdir(label_dir)
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file)) as current_file:
            labels = current_file.readlines()
        for label in labels:
            obj_name = label.split(' ')[0]
            try:
                obj_count_dict[obj_name] += 1
            except NameError and KeyError:
                continue
    return obj_count_dict

def append_truths_to_predictions(label_file, score_list):
    with open(label_file, 'r') as labels:
        lines = labels.readlines()
        for line in lines:
            if line == '\n':
                lines.remove('\n')
        new_content = ''
        for index, line in enumerate(lines):
            if line == '' or line == '\n':
                continue
            if score_list[index] is None:
                new_content += line
                continue

            field_list = line.split(' ')
            if len(field_list) < 17:
                if len(field_list) < 16:
                    field_list.append('0')
                field_list.append('0')
            field_list[16] = f'{score_list[index]}'
            if '\n' not in field_list:
                field_list.append('\n')
            new_content += label_list_to_label_line(field_list)

    with open(label_file, 'w') as labels:
        labels.write(new_content)
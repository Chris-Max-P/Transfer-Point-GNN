import pye57
import numpy as np
import platform
import os
from tqdm import tqdm

platform = platform.system()
if platform == 'Windows':
    splitter = '\\'
    labels = 'F:\\TransferPoint-GNN\\MUT_unsplitted\\labels'
    labels_splitted = 'F:\\TransferPoint-GNN\\MUT\\labels'
    tunnel = 'F:\\TransferPoint-GNN\\MUT_unsplitted\\tunnel'
    tunnel_splitted = 'F:\\TransferPoint-GNN\\MUT\\tunnel'
else:
    splitter = '/'
    labels = '/home/stud/paul/dataset/MUT_unsplitted/labels'
    labels_splitted = '/home/stud/paul/dataset/MUT/labels'
    tunnel = '/home/stud/paul/dataset/MUT_unsplitted/tunnel'
    tunnel_splitted = '/home/stud/paul/dataset/MUT/tunnel'


def split_e57(e57_file, split_labels=True):
    file_name = e57_file.split(f'tunnel{splitter}')[1].split('.')[0]
    # read e57 =================================
    e57 = pye57.E57(e57_file)
    header = e57.get_header(0)
    translation = header.translation
    data = e57.read_scan_raw(0)
    assert isinstance(data["cartesianX"], np.ndarray)
    assert isinstance(data["cartesianY"], np.ndarray)
    assert isinstance(data["cartesianZ"], np.ndarray)
    assert isinstance(data["intensity"], np.ndarray)
    intensity = data["intensity"].reshape(-1,1)
    x = data["cartesianX"]
    y = data["cartesianY"]
    z = data["cartesianZ"]
    points = np.vstack((x, y, z)).T
    cloud_complete = np.hstack((points, intensity))
    # split e57 ===============================
    cloud_complete = cloud_complete[cloud_complete[:,1].argsort()]

    tar_size = 3500000  # ca 5 m
    overlap = 1750000  # ca. 2.5 m
    current_cloud_size_in_points = cloud_complete.shape[0]

    index = 0
    cloud_list = []
    border_values = []
    y_translation = translation[1]
    while index <= current_cloud_size_in_points:
        border = min(index + tar_size, current_cloud_size_in_points)
        border_values.append(cloud_complete[border-1, 1] + y_translation)
        cloud = cloud_complete[index:border]
        cloud_list.append(cloud.T)
        index += overlap

    # save cloud to files ==============================
    for index, small_cloud in enumerate(cloud_list):
        small_cloud_dict = {
            'cartesianX': small_cloud[0, :],
            'cartesianY': small_cloud[1, :],
            'cartesianZ': small_cloud[2, :],
            'intensity': small_cloud[3, :]
        }

        e57_path = f"{os.path.join(tunnel_splitted, file_name)}({index}).e57"
        e57_write = pye57.E57(e57_path, mode='w')
        e57_write.write_scan_raw(small_cloud_dict, translation=translation)
    e57_write = pye57.E57(os.path.join(tunnel_splitted,'empty file'), mode='w') #emptying buffer
# split labels ================================
    if split_labels:
        split_label_file(file_name, border_values)


def split_label_file(file_name, border_values):
    label_file = os.path.join(labels, file_name + '.txt')
    contents = ['' for _ in range(len(border_values))]
    with open(label_file) as file:
        lines = file.readlines()
    for line in lines:
        fields = line.split(' ')
        y = float(fields[11])
        for index, border in enumerate(border_values):
            if y <= border:
                contents[index] += line
                break
    for index, content in enumerate(contents):
        new_label_path = os.path.join(labels_splitted, file_name)
        with open(f'{new_label_path}({index}).txt', mode='w') as file:
            file.write(content)

def split_splits(split_dir, labels_dir):
    splits = os.listdir(split_dir)
    labels = os.listdir(labels_dir)
    labels.sort()
    for split_file in splits:
        current_file = os.path.join(split_dir, split_file)
        new_content = ''
        with open(current_file) as current_split_file:
            old_label_names = current_split_file.readlines()

        for old_label in old_label_names:
            file_name = old_label.replace('.txt\n','')
            if file_name + '(0).txt' in labels:
                index = labels.index(file_name + '(0).txt')
                while file_name in labels[index]:
                    new_content += labels[index] + '\n'
                    index += 1
        new_split_file_name = current_file.replace('.txt', '_splitted.txt')
        with open(new_split_file_name, mode='w') as new_split_file:
            new_split_file.write(new_content)


def split_all_clouds_and_labels(split_labels=True, splits=True):
    tunnels = os.listdir(tunnel)
    splitted_tunnels = os.listdir(tunnel_splitted)
    already_splitted = [file_name.replace('(0)', '') for file_name in splitted_tunnels
                        if '(0)' in file_name]
    to_split = list(set(tunnels) - set(already_splitted))
    bad_files = []
    for _, pc in tqdm(enumerate(to_split)):
        e57_pc = os.path.join(tunnel, pc)
        print(f'Splitting {pc}')
        try:
            split_e57(e57_pc, split_labels=split_labels)
        except pye57.libe57.E57Exception as x:
            bad_files.append(pc)
            print(x)
    if splits:
        split_splits()
    print("Did not work for: ")
    print(str(bad_files))

split_all_clouds_and_labels(split_labels=False, splits=False)






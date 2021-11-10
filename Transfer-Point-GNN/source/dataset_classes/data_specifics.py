all_class_names_mut = ['Background', 'Anschlusskasten', 'Antenne',
                       'Beleuchtung', 'Fernsprecher', 'Gelaender',
                       'Laufweg', 'Leiter', 'Magnet', 'Schild',
                       'Signal', 'Spiegeltuer', 'SR-Stab(IST-Lage)',
                       'SR-Stab(SOLL-Lage)', 'Uhr', 'DontCare']

all_class_names_kitti = ['car', 'pedestrian', 'cyclist', 'van',
                         'person_sitting', 'car', 'tractor', 'trailer']

class_to_name_dict_mut_all = {0: 'Background', 1: 'Anschlusskasten', 2: 'Antenne',
                              3: 'Beleuchtung', 4: 'Fernsprecher', 5: 'Gelaender',
                              6: 'Laufweg', 7: 'Leiter', 8: 'Magnet', 9: 'Schild',
                              10: 'Signal', 11: 'Spiegeltuer', 12: 'SR-Stab(IST-Lage)',
                              13: 'SR-Stab(SOLL-Lage)', 14: 'Uhr', 15: 'DontCare'}

class_to_name_dict_mut = {0: 'SR-Stab',
                          1: 'Laufweg',
                          2: 'Magnet',
                          3: 'Schild',
                          4: 'Leiter',
                          5: 'SR-Stab(IST-Lage)',
                          6: 'SR-Stab(SOLL-Lage)',
                          7: 'Signal'}

class_to_name_dict_kitti = {
    0: 'Car',
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Van',
    4: 'Person_sitting',
    5: 'car',
    6: 'tractor',
    7: 'trailer',
}

median_object_size_map = {
    # l, h, w
    'Cyclist': (1.76, 1.75, 0.6),
    'Van': (4.98, 2.13, 1.88),
    'Tram': (14.66, 3.61, 2.6),
    'Car': (3.88, 1.5, 1.63),
    'Misc': (2.52, 1.65, 1.51),
    'Pedestrian': (0.88, 1.77, 0.65),
    'Truck': (10.81, 3.34, 2.63),
    'Person_sitting': (0.75, 1.26, 0.59),

    # sizes double
    'Anschlusskasten': (0.84, 0.74, 0.54),
    'Antenne': (0.2, 0.4, 0.4),
    'Beleuchtung': (1.44, 0.2, 0.24),
    'Fernsprecher': (0.8, 3.8, 0.8),
    'Gelaender': (3.0, 4.0, 0.1),
    'Laufweg': (1.6, 0.4, 2.4),
    'Leiter': (0.8, 5.0, 0.5),
    'Magnet': (0.6, 0.5, 1.3),
    'Schild': (0.8, 0.6, 0.4),
    'Signal': (0.5, 1.8, 0.5),
    'Spiegeltuer': (0.2, 4, 2),
    'SR-Stab(IST-Lage)': (0.1, 2, 1),
    'SR-Stab(SOLL-Lage)': (0.1, 2, 0.1),
    'SR-Stab': (0.1, 2, 0.1),
    'Uhr': (0.6, 1.2, 1.2)
    # 'DontCare': (-1.0, -1.0, -1.0)
}
""" sizes simple
'Anschlusskasten': (0.42, 0.37, 0.27),
'Antenne': (0.1, 0.2, 0.2),
'Beleuchtung': (0.71, 0.1, 0.11),
'Fernsprecher': (0.4, 1.9, 0.4),
'Gelaender': (1.5, 2, 0.05),
'Laufweg': (0.8, 0.2, 1.2),
'Leiter': (0.4, 2.5, 0.25),
'Magnet': (0.3, 0.25, 0.65),
'Schild': (0.4, 0.3, 0.2),
'Signal': (0.25, 0.9, 0.25),
'Spiegeltuer': (0.1, 2, 1),
'SR-Stab(IST-Lage)': (0.05, 1, 0.5),
'SR-Stab(SOLL-Lage)': (0.05, 1, 0.05),
'SR-Stab': (0.05, 1, 0.05),
'Uhr': (0.3, 0.6, 0.6)
"""
# 1627 Cyclist mh=1.75; mw=0.6; ml=1.76;
# 2914 Van mh=2.13; mw=1.88; ml=4.98;
# 511 Tram mh=3.61; mw=2.6; ml=14.66;
# 28742 Car mh=1.5; mw=1.63; ml=3.88;
# 973 Misc mh=1.65; mw=1.51; ml=2.52; voxelnet
# 4487 Pedestrian mh=1.77; mw=0.65; ml=0.88;
# 1094 Truck mh=3.34; mw=2.63; ml=10.81;
# 222 Person_sitting mh=1.26; mw=0.59; ml=0.75;
# 11295 DontCare mh=-1.0; mw=-1.0; ml=-1.0;

from_scratch_list = ['from_scratch_rot']
finetune_different_objects_list = ['few_labels_rot', 'many_labels_rot']
finetune_posts_list = ['posts_rot_128', 'posts_rot_256', 'posts_rot_512', 'posts_rot_786', 'posts_rot_1024']

all = {'Background': 0, 'Anschlusskasten': 1, 'Antenne': 3,
       'Beleuchtung': 5, 'Fernsprecher': 7, 'Gelaender': 9,
       'Laufweg': 11, 'Leiter': 13, 'Magnet': 15, 'Schild': 17,
       'Signal': 19, 'Spiegeltuer': 21, 'SR-Stab(IST-Lage)': 23,
       'SR-Stab(SOLL-Lage)': 25, 'Uhr': 27, 'DontCare': 29
       }

mostly_rot_sym = {
    'Background': 0,
    'Anschlusskasten': 1,
    'Antenne': 3,
    'Leiter': 5,
    'Magnet': 7,
    'Schild': 9,
    'SR-Stab': 11,
    'Stützen': 13,
    'Uhr': 15,
    'DontCare': 17
}

rot_sym = {
    'Background': 0,
    'Anschlusskasten': 1,
    'Schild': 2,
    'SR-Stab': 3,
    'Uhr': 4,
    'DontCare': 5
}

rot_sym2 = {
    'Background': 0,
    'Laufweg': 1,
    'SR-Stab': 3,
    'DontCare': 5
}


posts_rot = {  # 2 is for horizontal vs. vertical from KITTI impl
    'Background': 0,
    'SR-Stab': 1,
    'DontCare': 3
}

few_labels_rot = {
    'Background': 0,
    'Schild': 1,
    'Signal': 3,
    'DontCare': 5
}

many_labels_rot = {
    'Background': 0,
    'SR-Stab': 1,
    'Magnet': 3,
    'DontCare': 5
}

from_scratch_rot = {
    'Background': 0,
    'SR-Stab': 1,
    'Magnet': 3,
    'Schild': 5,
    'Signal': 7,
    'DontCare': 9
}

def get_label_map(label_method):
    # MUT label_methods
    if label_method == 'all':
        label_map = all
    elif label_method == 'rot_sym':
        label_map = rot_sym
    elif label_method == 'rot_sym2':
        label_map = rot_sym2
    elif label_method == 'mostly_rot_sym':
        label_map = mostly_rot_sym

    # MUT tested label_methods
    elif 'posts_rot' in label_method:
        label_map = posts_rot
    elif label_method == 'few_labels_rot':
        label_map = few_labels_rot
    elif 'many_labels_rot' in label_method:
        label_map = many_labels_rot
    elif label_method == 'from_scratch_rot':
        label_map = from_scratch_rot

    # KITTI label_methods
    elif label_method == 'yaw':
        label_map = {
            'Background': 0,
            'Car': 1,
            'Pedestrian': 3,
            'Cyclist': 5,
            'DontCare': 7
        }
    elif label_method == 'Car' or label_method == 'KITTI_train_test':
        label_map = {
            'Background': 0,
            'Car': 1,
            'DontCare': 3
        }
    elif label_method == 'Pedestrian_and_Cyclist':
        label_map = {
            'Background': 0,
            'Pedestrian': 1,
            'Cyclist': 3,
            'DontCare': 5
        }
    elif label_method == 'alpha':
        label_map = {
            'Background': 0,
            'Car': 1,
            'Pedestrian': 3,
            'Cyclist': 5,
            'DontCare': 7
        }
    return label_map


# second of each label is for horizontal vs. vertical (-> dataset.assign_classaware_label_to_points)
def label_map_to_double_list(label_map):
    label_list = []
    max_value = label_map.get('DontCare')
    for (key, value) in label_map.items():
        label_list.append(key)
        if 0 < value < max_value:
            label_list.append(key)
    return label_list

def label_map_to_simple_list(label_map):
    label_list = []
    max_value = label_map.get('DontCare')
    for (key, value) in label_map.items():
        if 0 < value < max_value:
            label_list.append(key)
    return label_list


def get_class_to_name_dict(dataset):
    class_to_name_dict = {}
    if dataset == 'MUT':
        class_to_name_dict = class_to_name_dict_mut
    elif dataset == 'KITTI':
        class_to_name_dict = class_to_name_dict_kitti
    return class_to_name_dict


def get_all_class_names(dataset):
    all_class_names = []
    if dataset == 'MUT':
        all_class_names = all_class_names_mut
    elif dataset == 'KITTI':
        all_class_names = all_class_names_kitti
    return all_class_names
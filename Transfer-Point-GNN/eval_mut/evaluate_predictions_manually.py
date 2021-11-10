import random

from source.dataset_classes.dataset import Dataset
from util.paths_and_data import *
from source.dataset_classes.dataset import get_label, box3d_to_cam_points
from source.data_preprocessing.label_file_operations import label_dict_to_label_line, count_objects_in_split, append_truths_to_predictions
import json
import os
from eval_mut.statistics import variance


# manual evaluation =========================================
from util_point_gnn.config_util import load_config


def create_dataset(DATASET, point_cloud_dir, pred_dir, file_list=None):
    if file_list is None:
        file_list = [file_name.replace('.txt', '') for file_name in os.listdir(pred_dir)]
    if DATASET == 'KITTI':
        point_cloud_format = '.bin'
        is_raw = False
    elif DATASET == 'MUT':
        point_cloud_format = '.e57'
        is_raw = True


    dataset = Dataset(
        '',
        point_cloud_dir,
        '',
        pred_dir,
        point_cloud_format,
        num_classes=4,
        is_raw=is_raw,
        file_list=file_list
    )
    return dataset

def objects_exist_in_frame(dataset, frame_idx, obj_list=None):
    box_label_list = dataset.get_label(frame_idx)
    for label_dict in box_label_list:
        if label_dict['name'] in obj_list:
            return True
    return False

def rate_object_BB_from_label(dataset, frame_idx, obj_list=None, evaluate=True):
    if obj_list is None or objects_exist_in_frame(dataset, frame_idx, obj_list):
        points_xyz = dataset.get_velo_points(frame_idx, debug=True).xyz
    else:
        return
    # 0 -> 11, 1 -> 13, 2 -> 15
    # 3 -> 3, 4 -> 4, 5 -> 6, 6 -> 8
    box_label_list = dataset.get_label(frame_idx)

    score_list = []
    for label_dict in box_label_list:
        if evaluate and 'eval_result' in label_dict:
            score_list.append(None)
            continue
        if obj_list is None:
            dataset.sel_xyz_in_box3d(label_dict, points_xyz, debug=True)
        elif label_dict['name'] in obj_list:
            dataset.sel_xyz_in_box3d(label_dict, points_xyz, debug=True)

        if evaluate:
            correct = input('Label correct?')
            if correct == '' or correct == 'y' or correct == '1':
                score = 1
            else:
                score = 0
            score_list.append(score)
    if evaluate:
        if all(score is None for score in score_list):
            return
        dataset.append_truths_to_predictions(frame_idx, score_list)


def iterate_through_predictions_for_evaluation(dataset, obj_list=None, evaluate=True):
    for frame_idx in range(dataset.num_files):
        print(f'eval {frame_idx} of {dataset.num_files}')
        rate_object_BB_from_label(dataset, frame_idx, obj_list=obj_list, evaluate=evaluate)

def evaluate_label_file(dataset, frame_idx):
    box_label_list = dataset.get_label(frame_idx)
    num_pred = 0
    TP = 0
    FP = 0
    for label_dict in box_label_list:
        num_pred += 1
        eval_result = label_dict['eval_result']
        if eval_result == 0:
            FP += 1
        elif eval_result == 1:
            TP += 1

    return num_pred, TP, FP


def iterate_randomly_for_evaluation(dataset, sample_size):
    num_pred = 0
    TP = 0
    FP = 0
    frame_list = list(range(dataset.num_files))
    random_sample = random.sample(frame_list, sample_size)
    for frame_idx in random_sample:
        rate_object_BB_from_label(dataset, frame_idx)
        add_num_pred, add_TP, add_FP, FPs = evaluate_label_file(dataset, frame_idx)
        num_pred += add_num_pred
        TP += add_TP
        FP += add_FP
    precision = TP / (TP + FP)
    return precision

def statistical_random_evaluation(dataset, sample_size, repeat=10):
    precisions = [iterate_randomly_for_evaluation(dataset, sample_size) for _ in range(repeat)]
    mittelwert, var, deviation = variance(precisions)
    return mittelwert, var, deviation

def evaluate_predicted_labels(dataset):
    num_pred = 0
    TP = 0
    FP = 0
    for frame_idx in range(dataset.num_files):
        add_num_pred, add_TP, add_FP = evaluate_label_file(dataset, frame_idx)
        num_pred += add_num_pred
        TP += add_TP
        FP += add_FP
    print('Manual evaluation yields:')
    print(f'num_pred: {num_pred}')
    print(f'TP: {TP}')
    print(f'FP: {FP}')
    precision = TP / (TP + FP)
    print(f'Precision: {precision}')
    return num_pred, TP, FP, precision


def evaluate_manually(pred_dir, obj_list=None, evaluate=True, file_list=None, random=None):
    """
    :param pred_dir: Prediction directory to evaluate
    :param obj_list: None to show all objects,
                    list of objects to show only specific ones
    :param evaluate: True if predictions are to be evaluated manually
    """

    dataset = create_dataset('MUT', tunnel_dir, pred_dir, file_list=file_list)
    if random is None:
        iterate_through_predictions_for_evaluation(dataset, obj_list=obj_list, evaluate=evaluate)
        return evaluate_predicted_labels(dataset)
    else:
        return statistical_random_evaluation(dataset, random)



# rank-based evaluation =========================================================
def get_pred_list(pred_dir):
    predictions = []
    for label_file in os.listdir(pred_dir):
        for label_list in get_label(os.path.join(pred_dir, label_file)):
            predictions.append((label_file, label_list))
    return predictions

def save_predictions(pred_tuple_list, save_path):
    for label_file, label_dict in pred_tuple_list:
        current_file_path = os.path.join(save_path, label_file)
        label_line = label_dict_to_label_line(label_dict)
        with open(current_file_path, 'a+') as target_file:
            if label_line not in target_file.readlines():
                target_file.write(label_line)

def get_top_x_predictions(pred_path, x, start_at=0):
    pred_list = get_pred_list(pred_path)
    pred_list.sort(key=lambda tup: tup[1]['score'], reverse=True)
    return pred_list[start_at:x]

def rank_based_prescision_score():
    # wighted precision:
        # norm scores
        # sum(prod(score*truth)) / sum(score)
    pass

def evaluate_top_x_predictions_manually(checkpoint, x):
    save_dir = os.path.join('../checkpoints', checkpoint, 'manual_eval', 'top_' + str(x))
    if not (os.path.exists(save_dir)):
        os.mkdir(save_dir)
    num_already_saved = len(os.listdir(save_dir))
    if num_already_saved < x:
        pred_path = os.path.join(results_dir, checkpoint)
        top_x = get_top_x_predictions(pred_path, x, start_at=num_already_saved)
        save_predictions(top_x, save_dir)

    file_list = [file_name.replace('.txt', '') for file_name in os.listdir(save_dir)]
    num_pred, TP, FP, precision = evaluate_manually(save_dir, file_list=file_list)

    result_dict = {
        'num_pred_analyzed': num_pred,
        'TP': TP,
        'FP': FP,
        'precision': precision
    }
    with open(os.path.join('../checkpoints', checkpoint, f'top_{x}_stats.json'), 'w') as results:
        results.write(json.dumps(result_dict))


def is_center_of_first_in_box_of_second(first_box_dict, second_box_dict, expand_factor=(1.0, 1.0, 1.0)):
    box = box3d_to_cam_points(second_box_dict, expand_factor)
    x = first_box_dict['x3d']
    y = first_box_dict['y3d']
    z = first_box_dict['z3d']

    x_in_box = box[0][0][0] >= x >= box[0][2][0] # right / left
    y_in_box = box[0][0][1] >= y >= box[0][4][1] # up / down
    z_in_box = box[0][0][2] >= z >= box[0][1][2] # front / back

    return x_in_box and y_in_box and z_in_box


def compare_pred_with_ground_truths(pred_dir, gt_dir, expand_factor=(1.0, 1.0, 1.0), append_eval=False):
    FNs = []
    FPs = [] # FP + FFP
    TPs = []
    for filename in os.listdir(pred_dir):
        pred_file = os.path.join(pred_dir, filename)
        gt_file = os.path.join(gt_dir, filename)

        gt_label_dict_list = get_label(gt_file)
        pred_label_dict_list = get_label(pred_file)

        score_list = []
        for pred_label_dict in pred_label_dict_list:

            TP = False
            for gt_label_dict in gt_label_dict_list:
                if gt_label_dict['name'] != 'SR-Stab':
                    continue
                if is_center_of_first_in_box_of_second(pred_label_dict, gt_label_dict, expand_factor=expand_factor):
                    TP = True
            if TP:
                TPs.append((filename, pred_label_dict))
                score_list.append(1)
            else:
                FPs.append((filename, pred_label_dict))
                score_list.append(0)
        if append_eval:
            append_truths_to_predictions(pred_file, score_list)

        for gt_label_dict in gt_label_dict_list:
            if gt_label_dict['name'] != 'SR-Stab':
                continue
            FN = True
            for pred_label_dict in pred_label_dict_list:
                if is_center_of_first_in_box_of_second(gt_label_dict, pred_label_dict, expand_factor=expand_factor):
                    FN = False
            if FN:
                FNs.append((filename, gt_label_dict))

    return FNs, FPs, TPs

def delete_doubled_predictions(pred_dir, expand_factor=(1.0, 1.0, 1.0)):
    # check for similarity
    delete_from = []
    for filename in os.listdir(pred_dir):
        pred_file = os.path.join(pred_dir, filename)

        pred_label_dict_list = get_label(pred_file)
        to_delete = []

        for pred_label_dict in pred_label_dict_list:
            pred_label_dict_list.remove(pred_label_dict)
            for pred_label_dict_2 in pred_label_dict_list:
                if is_center_of_first_in_box_of_second(pred_label_dict_2, pred_label_dict, expand_factor=expand_factor):
                    to_delete.append(label_dict_to_label_line(pred_label_dict_2))
        delete_from.append((pred_file, to_delete))
    # delete doubles
    for twople in delete_from:
        filename, to_delete = twople
        if len(to_delete) > 0:
            with open(filename, 'r') as file:
                lines = file.readlines()
                for line in to_delete:
                    line = line.replace('-1.0', '-1').replace('0.0 ', '0 ')
                    if line in lines:
                        lines.remove(line)
            with open(filename, 'w') as file:
                content = ''
                for line in lines:
                    content += line
                file.write(content)

def lists_overlap(list1, list2):
    overlap = []
    for item in list1:
        if item in list2:
            overlap.append(item)
    return overlap

def print_auto_eval_results(FNs, FPs, TPs, num_pred, num_gt):
    print('Auto eval yields:')
    print(f'{len(FPs)} FPs + {len(TPs)} TPs = {num_pred} predictions')
    print(f'{len(FNs)} FNs + {len(TPs)} TPs = {len(FNs) + len(TPs)} != {num_gt} ground truths')
    print('\n')

def auto_eval_predictions(pred_dir, checkpoint, append_eval=False):
    num_pred = len(get_pred_list(pred_dir))

    split_file_name = load_config(f'../configs/{checkpoint}_meta_config')['test_dataset']
    split_file = os.path.join(split_dir, split_file_name)
    obj_count_dict, num_files_in_split = count_objects_in_split(['SR-Stab'], split_file, rot_sym_label_dir)
    num_ground_truths = obj_count_dict['SR-Stab']

    #for i in range(1,4):
    i = 2
    print(f'With BB expanded by {i}:')
    j = float(i)
    FNs, FPs, TPs = compare_pred_with_ground_truths(pred_dir, rot_sym_label_dir, expand_factor=(j, j, j), append_eval=append_eval)
    assert len(lists_overlap(FNs, FPs)) == 0
    assert len(lists_overlap(FNs, TPs)) == 0
    assert len(lists_overlap(FPs, TPs)) == 0
    assert len(FPs) + len(TPs) == num_pred
    # assert TP + FN == num_gt

    print_auto_eval_results(FNs, FPs, TPs, num_pred, num_ground_truths)

    if not os.path.exists(FN_dir):
        save_predictions(FNs, FN_dir)

    return num_ground_truths, len(FNs), len(FPs)

def get_pred_results(pred_dir, checkpoint, save_dir):
    dataset = create_dataset('MUT', tunnel_dir, pred_dir)
    num_pred, TP, FP, precision = evaluate_predicted_labels(dataset)
    num_gt, FN, FPrated = auto_eval_predictions(pred_dir, checkpoint)

    FFP = FPrated - FP
    TP = TP + FFP
    precision = TP / (TP + FP)
    recall = TP / (FP + FN)

    pred_results = {
        'num_gt': num_gt,
        'num_pred': num_pred,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'FFP': FFP,
        'precision': precision,
        'recall': recall
    }

    with open(os.path.join(save_dir, 'pred_results.json'), 'w') as res:
        res.write(json.dumps(pred_results, indent=4))

# executed code ============================================
checkpoint = 'posts_rot_1024_1'
pred_dir = os.path.join(results_dir, checkpoint)
save_dir = os.path.join('../checkpoints', checkpoint, 'manual_eval')
FP_dir = os.path.join(save_dir, 'FPs')
FN_dir = os.path.join(save_dir, 'FNs')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(FP_dir)
    os.mkdir(FN_dir)


#delete_doubled_predictions(pred_dir, expand_factor=(3.0, 3.0, 3.0))

#auto_eval_predictions(pred_dir, checkpoint, append_eval=False)

#evaluate_manually(pred_dir)
get_pred_results(pred_dir, checkpoint, save_dir)

#evaluate_manually(FN_dir, evaluate=False)
#mittelwert, var, deviation = evaluate_manually(FP_dir, random=10)
#print(f'Mittelwert: {mittelwert}, Varianz: {var}, Standardabweichung: {deviation}')



"""# inf
point_cloud_dir = "F:\\TransferPoint-GNN\\MUT_mini\\tunnel"
prediction_dir = 'F:\\Results\\rot_sym6_finetune1_inf'
evaluate_manually(point_cloud_dir, prediction_dir)"""

"""# investigation of already known objects
point_cloud_dir = 'F:\\TransferPoint-GNN\\MUT_unsplitted\\tunnel'
label_dir = 'F:\\TransferPoint-GNN\\MUT_unsplitted\\labels'
#label_dir = "F:\\TransferPoint-GNN\\MUT\\rot_sym_labels"
obj_list = ['Uhr']
evaluate_predictions(point_cloud_dir, label_dir, evaluate=False)"""




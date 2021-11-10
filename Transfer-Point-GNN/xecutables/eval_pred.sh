#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name eval
#SBATCH --output /home/stud/paul/Results/eval%A.out
cd /home/stud/paul/Point-GNN.pytorch/kitti-eval

#paths setup
case "$1" in
  MUT | "")
        dataset_root_dir="/home/stud/paul/dataset/MUT"
        label_path_suffix="/labels"
        result_path_suffix="${2}"
        dataset="MUT";;
  KITTI)
        dataset_root_dir="/nfs/data3/paul/kitti/"
        label_path_suffix="labels/training/label_2"
        result_path_suffix="${2}"
        dataset="KITTI";;
esac

result_root_dir="/home/stud/paul/Results/"
label_path="${dataset_root_dir}${label_path_suffix}"
result_path="${result_root_dir}${result_path_suffix}/"
label_split_file="${dataset_root_dir}/3DOP_splits/val.txt"

#environment setup
source /home/stud/paul/anaconda3/etc/profile.d/conda.sh
conda activate eval

python evaluate.py evaluate --dataset=${dataset} --label_path=${label_path} --result_path=${result_path} --label_split_file=${label_split_file} --current_class=0 --coco=False

# $1: dataset
# $2: directory to evaluate
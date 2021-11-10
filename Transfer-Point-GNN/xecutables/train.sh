#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name fineT
#SBATCH --output /home/stud/paul/Results/finetune%A.out
#SBATCH -e /home/stud/paul/Results/finetune%A.err
#SBATCH --cpus-per-task=8
cd /home/stud/paul/Point-GNN.pytorch

checkpoint="$1"
case "$2" in
  MUT | "")
    dataset="MUT"
    dataset_root_dir="/home/stud/paul/dataset/MUT/"
    if [[ $1 == *"rot"* ]];
      then labels_dir="rot_sym_labels"
      else labels_dir="labels"
    fi
    echo "using $labels_dir for training"
    ;;
  KITTI)
    dataset='KITTI'
    dataset_root_dir="/nfs/data3/paul/kitti/"
    labels_dir="x"
    ;;
esac

#environment setup
source /home/stud/paul/anaconda3/etc/profile.d/conda.sh
conda activate tf2

python3 train.py ${checkpoint} ${dataset} --dataset_root_dir ${dataset_root_dir} --label_dir ${labels_dir}

# $1: checkpoint
# $2: dataset

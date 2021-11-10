#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name inf
#SBATCH --output /home/stud/paul/Results/inf%A.out
#SBATCH -e /home/stud/paul/Results/inf%A.err
#SBATCH --cpus-per-task=8
cd /home/stud/paul/Point-GNN.pytorch/

#environment setup
source /home/stud/paul/anaconda3/etc/profile.d/conda.sh
conda activate tf2

#paths setup
case "$2" in # (MUT, KITTI or MIX (for inference from KITTI model in MUT data))
  MUT | "")
        checkpoint_dir="./checkpoints/"${1}"/"
        dataset_root_dir="/home/stud/paul/dataset/MUT/"
        dataset="MUT";;
  KITTI)
        checkpoint_dir="./checkpoints/KITTI_test/"
        dataset_root_dir="/nfs/data3/paul/kitti/"
        dataset="KITTI";;
  MIX)
        checkpoint_dir="./checkpoints/inf_from_KITTI_on_MUT/"
        dataset_root_dir="/home/stud/paul/dataset/MUT/"
        dataset="MUT";;
esac

output_dir="/home/stud/paul/Results/"
jobname="${1}"

python3 inf.py ${dataset} ${checkpoint_dir} --dataset_root_dir ${dataset_root_dir} --output_dir ${output_dir} --jobname ${jobname}


# $1: checkpoint
# $2: dataset
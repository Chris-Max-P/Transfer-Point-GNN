#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name label-flow
#SBATCH --output /home/stud/paul/Results/label-flow-%A.out
#SBATCH -e /home/stud/paul/Results/label-flow-%A.err
#SBATCH --cpus-per-task=8
cd /home/stud/paul/Point-GNN.pytorch/xecutables

num_labels="$1"
num_layers="$2"

source="/home/stud/paul/Point-GNN.pytorch/checkpoints/posts_rot_${num_labels}_${num_layers}/"

for VARIABLE in 1 2 3 4
do
  target="/home/stud/paul/Point-GNN.pytorch/checkpoints/${num_labels}_${num_layers}_${VARIABLE}"
  mkdir "${target}"
  bash train.sh posts_rot_"${num_labels}"_"${num_layers}"
  mv "${source}" "${target}"
  mkdir "${source}"
done
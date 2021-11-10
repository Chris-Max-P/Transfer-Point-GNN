#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name eval
#SBATCH --output /home/stud/paul/Results/test-%A.out
#SBATCH -e /home/stud/paul/Results/test%A.err
#SBATCH --cpus-per-task=8
cd /home/stud/paul/Point-GNN.pytorch/xecutables

num_labels="$1"
num_layers="$2"

for VARIABLE in 1 2 3 4 5
do
  bash eval.sh posts_rot_"${num_labels}"_"${num_layers}" MUT test
done
#!/bin/bash

#SBATCH -J dqn_training
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output logs/dqn_training-%j.out  
#SBATCH --error logs/dqn_training-%j.err  

module load cuda

cd /home/bhu3/projects/WPI-DS551-Fall25/Project3

mode=train_dqn

total_frames=5000000
batch_size=32
train_freq=4
replay_buffer_size=1000000
replay_start_size=50000

target_update_freq=10000

gamma=0.99

eps_start=1.0
eps_end=0.1
eps_decay=1000000

uv run main.py --$mode \
  --total_frames $total_frames \
  --batch_size $batch_size \
  --train_freq $train_freq \
  --replay_buffer_size $replay_buffer_size \
  --replay_start_size $replay_start_size \
  --target_update_freq $target_update_freq \
  --gamma $gamma \
  --eps_start $eps_start \
  --eps_end $eps_end \
  --eps_decay $eps_decay
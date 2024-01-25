#!/bin/bash

# 224m41.472s
# 183m48.340s

_basedir=/home/kykim/dev/t2i-ft/sft

num_seeds=1
num_imgs_per_seed=10
reward_type=clip,blip,pickscore,imagereward
cqa_file=prompts.txt  # prompts.txt, qas.json

pyfile=${_basedir}/generate_train_samples.py

source ~/dev/venv/ddpo/bin/activate

i=0
# for rtype in clip blip imagereward pickscore mean_ensemble uw_ensemble; do
for dataset in bon_compose_10; do
  for rtype in imagereward pickscore mean_ensemble uw_ensemble; do
    outdir_base=${_basedir}/experiments/grid_search/${dataset}/${rtype}
    # b32_lr1e-05_kl0.0_es0.1_l10k_norm2_max1000
    for d in $(ls -d ${outdir_base}/b32_*); do
      dir_base=$(basename ${d})
      device=$(( i % 6 ))

      CUDA_VISIBLE_DEVICES=${device} accelerate launch ${pyfile} \
      --cqa_file=${_basedir}/datasets/${dataset}/${cqa_file} \
      --num_seeds=${num_seeds} --num_imgs_per_seed=${num_imgs_per_seed} \
      --outdir=${d} --lora_paths=all --reward_type=${rtype} &
      (( ++i ))
      if (( i % 6 == 0 )); then
        wait
        echo "Waited ${i}"
      fi
    done
  done
done

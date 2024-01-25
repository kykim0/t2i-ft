#!/bin/bash

# 526m54.247s
# 444m52.932s

# elite_samples=0.1
train_bs=4
train_steps=1000
metadata_filename=sft_train_data_ours.json
ckpt_steps=50
valid_steps=10000
metadata_limit=10000
_basedir=/home/kykim/dev/t2i-ft/sft
seed=0

pyfile=${_basedir}/train_text_to_image_lora_kl.py

source ~/dev/venv/ddpo/bin/activate

grad_accum_steps=8
# clip blip imagereward pickscore mean_ensemble uw_ensemble
i=0
for dataset in bon_count_10; do
  for kl_coeff in 0.0 0.25 0.5 0.75 1.0; do
    for lr in 1e-05; do
      for rtype in imagereward pickscore mean_ensemble uw_ensemble; do
        device=$(( i % 6 ))

        train_data_dir=${_basedir}/datasets/${dataset}
        batch_size=$(( $grad_accum_steps * $train_bs * 1 ))
        output_dir=${_basedir}/experiments/grid_search/${dataset}/${rtype}
        exp_dir=${output_dir}/b${batch_size}_lr${lr}_kl${kl_coeff}

        proj_name=t2i-ft-grid_search-${dataset}-${rtype}
        # --elite_samples=${elite_samples}

        [ ! -d $exp_dir ] && \
        echo CUDA_VISIBLE_DEVICES=${device} accelerate launch \
        --mixed_precision=fp16 ${pyfile} \
        --checkpointing_steps=${ckpt_steps} --learning_rate=${lr} --seed=${seed} --report_to=tensorboard \
        --train_data_dir=${train_data_dir} --train_batch_size=${train_bs} \
        --gradient_accumulation_steps=${grad_accum_steps} --max_train_steps=${train_steps} \
        --output_dir=${output_dir} --validation_steps=${valid_steps} \
        --kl_coeff=${kl_coeff} --reward_type=${rtype} \
        --tracker_project_name=${proj_name} \
        --train_metadata_filename=${metadata_filename} \
        --train_metadata_limit=${metadata_limit} \
        --normalize &
        (( ++i ))
        if (( i % 6 == 0 )); then
          wait
          echo "Waited ${i}"
        fi
      done
    done
  done
done

# Multi-GPU run.
#
# CUDA_VISIBLE_DEVICES=${device} accelerate launch \
# --multi_gpu --mixed_precision=fp16 train_text_to_image_lora_kl.py \
# --checkpointing_steps=${ckpt_steps} --learning_rate=${lr} --seed=${seed} --report_to=tensorboard \
# --train_data_dir=${train_data_dir} --train_batch_size=${train_bs} \
# --gradient_accumulation_steps=${grad_accum_steps} --max_train_steps=${train_steps} \
# --output_dir=${output_dir} --validation_steps=${valid_steps} \
# --kl_coeff=${kl_coeff} --r_threshold=${r_threshold} \
# --num_validation_images=4 --tracker_project_name=${proj_name} \
# --train_metadata_filename=${metadata_filename} --train_metadata_limit=${metadata_limit}

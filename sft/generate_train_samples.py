"""Generate training samples for SFT of T2I models.

Example usage:
  $ CUDA_VISIBLE_DEVICES=0,1 accelerate launch generate_train_samples.py \
    --cqa_file=datasets/dog_frog/qas.json --num_seeds=20 --num_imgs_per_seed=5
"""

import argparse
import collections
import errno
import glob
import itertools
import json
import os

from accelerate import PartialState
from diffusers import DiffusionPipeline
import torch
from tqdm.auto import tqdm

import rewards


state = PartialState()


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cqa_file', type=str, default='')
  parser.add_argument('--outdir', type=str, default=None)
  parser.add_argument('--num_seeds', type=int, default=20)
  parser.add_argument('--num_imgs_per_seed', type=int, default=1)
  parser.add_argument('--reward_type', type=str, default='vqa')
  parser.add_argument('--split_captions', type=bool, default=True)
  parser.add_argument('--model_name', type=str,
                      default='stabilityai/stable-diffusion-2-1')
  parser.add_argument('--revision', type=str, default=None)
  parser.add_argument('--lora_paths', type=str, default=None)
  parser.add_argument('--metadata_filename', type=str,
                      default='sft_train_data.json')
  args = parser.parse_args()
  return args


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def image_filename(pidx, seed, iidx, imgdir=None):
  if imgdir:
    return os.path.join(imgdir, f'image_{pidx}_{seed}_{iidx}.jpg')
  return os.path.join(f'image_{pidx}_{seed}_{iidx}.jpg')


def generate_images(args, all_captions, c_to_idx, imgdir, lora_path=None):
  """Generates images using multiple GPUs in parallel."""
  model_name = args.model_name
  dtype = torch.float16

  # Load a pre-trained model.
  pipe = DiffusionPipeline.from_pretrained(
      model_name,
      revision=args.revision,
      torch_dtype=dtype,
  )
  pipe.set_progress_bar_config(disable=True)
  if lora_path:
    pipe.unet.load_attn_procs(lora_path)

  device = state.device
  pipe.to(device)

  seeds = list(range(args.num_seeds))
  num_imgs_per_seed = args.num_imgs_per_seed

  def subroutine(captions, seeds):
    for seed in seeds:
      # Continue if all images for the seed already generated.
      filenames = [
          image_filename(c_to_idx[caption], seed, iidx, imgdir)
          for iidx in range(args.num_imgs_per_seed)
      ]
      if all(os.path.exists(fn) for fn in filenames): continue

      generator = torch.Generator(device).manual_seed(seed)
      img_results = pipe(captions, generator=generator).images
      for iidx, img_result in enumerate(img_results):
        filename = image_filename(c_to_idx[caption], seed, iidx, imgdir)
        img_result.save(filename)

  if args.split_captions:
    with state.split_between_processes(all_captions) as sub_captions:
      for caption in tqdm(sub_captions):
        captions = [caption] * num_imgs_per_seed
        print(f'({state.local_process_index}) Generating images for: '
              f'"{caption}" with seeds {seeds}')
        subroutine(captions, seeds)
  else:
    for caption in all_captions:
      captions = [caption] * num_imgs_per_seed
      # TODO(kykim): We tried shuffling the seed list before the split, but the
      # naive approach does not do it correctly. It appears that each process
      # does its own shuffling and takes its portion afterwards.
      with state.split_between_processes(seeds) as sub_seeds:
        print(f'({state.local_process_index}) Generating images for: '
              f'"{caption}" with seeds {sub_seeds}')
        subroutine(captions, tqdm(sub_seeds))

  del pipe
  torch.cuda.empty_cache()


def compute_rewards(args, paths, captions, image_names, cqas):
  """Computes rewards and returns the final data dict."""

  def subroutine(rtype, captions, image_paths):
    if rtype == 'vqa':
      return rewards.vqa_rewards(captions, image_paths, cqas)
    elif rtype == 'clip':
      return rewards.clip_score(captions, image_paths)
    elif rtype == 'blip':
      return rewards.blip_score(captions, image_paths)
    elif rtype == 'pickscore':
      return rewards.pick_score(captions, image_paths)
    elif rtype == 'imagereward':
      return rewards.image_reward(captions, image_paths)
    raise ValueError(f'Unsupported reward type: {rtype}')

  batch = 10
  reward_types = args.reward_type.split(',')
  with state.split_between_processes(paths) as sub_paths:
    for path in tqdm(sub_paths):
      # Skip the path if the metadata file exists.
      # TODO(kykim): Perhaps can do an extra check to only evaluate the ones we
      # have not already, but the reward computation is fast enough for now.
      metadata_filename = os.path.join(path, args.metadata_filename)
      if os.path.exists(metadata_filename): continue

      image_paths = [os.path.join(path, 'images', name)
                     for name in image_names]
      reward_dicts = collections.defaultdict(list)
      for idx in range((len(image_names) + batch - 1) // batch):
        sub_image_paths = image_paths[idx * batch:(idx + 1) * batch]
        sub_captions = captions[idx * batch:(idx + 1) * batch]
        for reward_type in reward_types:
          reward_dicts[reward_type].extend(
              subroutine(reward_type, sub_captions, sub_image_paths))

      data_dicts = []
      for idx, (caption, image_name) in enumerate(zip(captions, image_names)):
        data_dict = {
            'image': image_name,
            'caption': caption,
            'rewards': {
                'human': [
                    -1,  # Initialize human label as -1.
                ],
            }
        }
        for reward_type, all_rewards in reward_dicts.items():
          data_dict['rewards'][reward_type] = all_rewards[idx]
        data_dicts.append(data_dict)

      with open(metadata_filename, 'w') as f:
        json.dump(data_dicts, f, indent=4)

  torch.cuda.empty_cache()


def main():
  args = parse_args()

  # Read the json file of captions and QAs.
  cqa_file = args.cqa_file
  if not os.path.exists(cqa_file):
    print(f'File does not exist: {args.cqa_file}')
    return

  with open(cqa_file, 'r') as f:
    cqas = json.load(f)
  all_captions = list(cqas.keys())
  c_to_idx = {caption: idx for idx, caption in enumerate(all_captions)}

  # Generate images using either a pre-trained or fine-tuned model.
  basedir = os.path.dirname(cqa_file)
  outdir = args.outdir or basedir
  paths = [outdir]  # A hack to support the basic non-LoRA case.
  if args.lora_paths == 'all':
    paths = glob.glob(os.path.join(outdir, 'checkpoint', 'ckpt*'))
  elif args.lora_paths:
    paths = glob.glob(args.lora_paths)
  for idx, path in enumerate(paths):
    if state.is_main_process:
      print(f'[{idx+1}/{len(paths)}] Processing {path}')
    imgdir = os.path.join(path, 'images')
    lora_path = path if path != outdir else None
    mkdir_p(imgdir)
    generate_images(args, all_captions, c_to_idx, imgdir, lora_path=lora_path)

  # TODO(kykim): Somehow the VQA model gets replicated multiple times in the
  # main process memory. Run the reward computation using only a single GPU
  # until we fix the issue. This part is also fast enough.
  if not state.is_main_process: return

  # Compute rewards and write out to a json.
  image_names, captions = [], []
  for caption, seed, iidx in itertools.product(
      all_captions, range(args.num_seeds), range(args.num_imgs_per_seed)):
    image_names.append(image_filename(c_to_idx[caption], seed, iidx))
    captions.append(caption)
  compute_rewards(args, paths, captions, image_names, cqas)


if __name__ == '__main__':
  main()

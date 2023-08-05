"""Generate training samples for SFT of T2I models.

Example usage:
  $ CUDA_VISIBLE_DEVICES=1 accelerate launch generate_train_samples.py \
    --cqa_file=datasets/dog_frog/qas.json --num_seeds=20
"""

import argparse
import errno
import itertools
import json
import os

from accelerate import PartialState
from diffusers import DiffusionPipeline
import numpy as np
import torch
from tqdm.auto import tqdm

import rewards


state = PartialState()


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cqa_file', type=str, default='')
  # TODO(kykim): Support commandline prompts.
  parser.add_argument('--outdir', type=str, default=None)
  parser.add_argument('--num_seeds', type=int, default=20)
  parser.add_argument('--num_imgs_per_seed', type=int, default=1)
  parser.add_argument('--model_name', type=str,
                      default='stabilityai/stable-diffusion-2-1')
  parser.add_argument('--lora_path', type=str, default=None)
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


def image_filename(pidx, seed, iidx, imgdir):
  return os.path.join(imgdir, f'image_{pidx}_{seed}_{iidx}.png')


def generate_images(args, all_captions, c_to_idx, imgdir):
  """Generates images using multiple GPUs in parallel."""
  model_name = args.model_name
  dtype = torch.float16

  # Load a pre-trained model.
  pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
  pipe.set_progress_bar_config(disable=True)
  if args.lora_path:
    pipe.unet.load_attn_procs(args.lora_path)

  device = state.device
  pipe.to(device)

  num_imgs_per_seed = args.num_imgs_per_seed
  num_seeds = args.num_seeds
  seeds = list(range(num_seeds))
  for caption in all_captions:
    print(f'Generating images for: {caption}')
    captions = [caption] * num_imgs_per_seed
    with state.split_between_processes(seeds) as sub_seeds:
      # Continue if all images in the batch already generated.
      filenames = [
          image_filename(c_to_idx[caption], seed, iidx, imgdir)
          for iidx, seed in itertools.product(
              range(num_imgs_per_seed), sub_seeds)
      ]
      if all(os.path.exists(fn) for fn in filenames): continue

      for seed in tqdm(sub_seeds):
        generator = torch.Generator(device).manual_seed(seed)
        img_results = pipe(captions, generator=generator).images
        for iidx, img_result in enumerate(img_results):
          filename = image_filename(c_to_idx[caption], seed, iidx, imgdir)
          img_result.save(filename)

  state.wait_for_everyone()
  del pipe
  torch.cuda.empty_cache()


def compute_rewards(captions, image_paths, cqas):
  """Computes rewards and returns the final data dict."""
  vqa_rs = rewards.vqa_rewards(captions, image_paths, cqas)
  final_data_dicts = []
  for caption, image_path, vqa_r in zip(captions, image_paths, vqa_rs):
    image = os.path.basename(image_path)
    final_data_dicts.append({
        'image': image,
        'caption': caption,
        'rewards': {
            'human': -1,   # Initialize human label as -1.
            'vqa': vqa_r,
        }
    })
  return final_data_dicts


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
  subfolder = (f'images_{os.path.basename(args.lora_path)}'
               if args.lora_path else 'images')
  imgdir = os.path.join(outdir, subfolder)
  mkdir_p(imgdir)
  generate_images(args, all_captions, c_to_idx, imgdir)

  # Only the main process needs to run the rest of the logic.
  if not state.is_main_process: return

  # Compute rewards and write out to a json.
  image_paths, captions = [], []
  for caption, seed, iidx in itertools.product(
        all_captions, range(args.num_seeds), range(args.num_imgs_per_seed)):
    image_paths.append(image_filename(c_to_idx[caption], seed, iidx, imgdir))
    captions.append(caption)
  data_dicts = compute_rewards(captions, image_paths, cqas)
  # data_dicts = sorted(data_dicts, key=lambda d: d['image'])
  metadata_filename = args.metadata_filename
  if args.lora_path:
    name, ext = metadata_filename.split('.')
    metadata_filename = f'{name}_{os.path.basename(args.lora_path)}.{ext}'
  with open(os.path.join(outdir, metadata_filename), 'w') as f:
    json.dump(data_dicts, f, indent=4)


if __name__ == '__main__':
  main()

"""Generate training samples for SFT of T2I models.

Example usage:
  $ CUDA_VISIBLE_DEVICES=0,1 accelerate launch generate_train_samples.py \
    --cqa_file=datasets/dog_frog/qas.json --num_seeds=20 --num_imgs_per_seed=5
"""

import argparse
import errno
import glob
import itertools
import json
import os

from accelerate import PartialState
from diffusers import DiffusionPipeline
from tifascore.vqa_models import VQAModel
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
    return os.path.join(imgdir, f'image_{pidx}_{seed}_{iidx}.png')
  return os.path.join(f'image_{pidx}_{seed}_{iidx}.png')


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
  for caption in all_captions:
    captions = [caption] * num_imgs_per_seed
    # TODO(kykim): We tried shuffling the seed list before the split, but the
    # naive approach does not do it correctly. It appears that each process
    # does its own shuffling and takes its portion afterwards.
    with state.split_between_processes(seeds) as sub_seeds:
      print(f'({state.local_process_index}) Generating images for: '
            f'"{caption}" with seeds {sub_seeds}')
      for seed in tqdm(sub_seeds):
        # Continue if all images for the seed already generated.
        filenames = [
            image_filename(c_to_idx[caption], seed, iidx, imgdir)
            for iidx in range(num_imgs_per_seed)
        ]
        if all(os.path.exists(fn) for fn in filenames): continue
        generator = torch.Generator(device).manual_seed(seed)
        img_results = pipe(captions, generator=generator).images
        for iidx, img_result in enumerate(img_results):
          filename = image_filename(c_to_idx[caption], seed, iidx, imgdir)
          img_result.save(filename)

  state.wait_for_everyone()
  del pipe
  torch.cuda.empty_cache()


def compute_rewards(args, paths, captions, image_names, cqas):
  """Computes rewards and returns the final data dict."""
  vqa_model = VQAModel('mplug-large')

  # TODO(kykim): Perhaps can do an extra check to only evaluate the ones we
  # have not already, but the reward computation is fast enough for now.
  with state.split_between_processes(paths) as sub_paths:
    for path in tqdm(sub_paths):
      # Skip the path if the metadata file exists.
      metadata_filename = os.path.join(path, args.metadata_filename)
      if os.path.exists(metadata_filename): continue

      image_paths = [os.path.join(path, 'images', name)
                     for name in image_names]
      vqa_rs = rewards.vqa_rewards(captions, image_paths, cqas, vqa_model)

      data_dicts = []
      for caption, image_name, vqa_r in zip(captions, image_names, vqa_rs):
        data_dicts.append({
            'image': image_name,
            'caption': caption,
            'rewards': {
                'human': -1,   # Initialize human label as -1.
                'vqa': vqa_r,
            }
        })

      with open(metadata_filename, 'w') as f:
        json.dump(data_dicts, f, indent=4)

  state.wait_for_everyone()
  del vqa_model
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
  paths = ['']  # A hack to support the basic pre-trained case.
  if args.lora_paths == 'all':
    paths = glob.glob(os.path.join(outdir, 'checkpoint', 'ckpt*'))
  elif args.lora_paths:
    paths = glob.glob(args.lora_paths)
  for idx, path in enumerate(paths):
    if state.is_main_process:
      print(f'[{idx+1}/{len(paths)}] Processing {path}')
    imgbase = path if path else outdir
    imgdir = os.path.join(imgbase, 'images')
    mkdir_p(imgdir)
    generate_images(args, all_captions, c_to_idx, imgdir, lora_path=path)

  # Compute rewards and write out to a json.
  image_names, captions = [], []
  for caption, seed, iidx in itertools.product(
      all_captions, range(args.num_seeds), range(args.num_imgs_per_seed)):
    image_names.append(image_filename(c_to_idx[caption], seed, iidx))
    captions.append(caption)
  compute_rewards(args, paths, captions, image_names, cqas)


if __name__ == '__main__':
  main()

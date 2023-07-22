"""Generate training samples for SFT of T2I models.

Example usage:
  $ CUDA_VISIBLE_DEVICES=0,1 accelerate launch generate_train_samples.py
"""

import argparse
import errno
import itertools
import json
import os

from accelerate import PartialState
from diffusers import StableDiffusionPipeline
import torch

import rewards


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cqa_file', type=str, default='')
  # TODO(kykim): Support commandline prompts.
  parser.add_argument('--basedir', type=str, default='./')
  parser.add_argument('--num_seeds', type=int, default=20)
  # Be sure to adjust it depending on the no. of processes and images.
  parser.add_argument('--per_batch_size', type=int, default=16)
  parser.add_argument('--model_id', type=str,
                      default='runwayml/stable-diffusion-v1-5')
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


def image_filename(pidx, seed):
  """Returns an image filename."""
  return f'image_{pidx}_{seed}.png'


def generate_images(args, all_captions, c_to_idx):
  """Generates images using multiple GPUs in parallel."""
  model_id = args.model_id
  
  # Load a pre-trained model.
  pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                 torch_dtype=torch.float16)
  
  distributed_state = PartialState()
  num_procs = distributed_state.num_processes
  pipe.to(distributed_state.device)

  def img_fn(pidx, seed):
    return os.path.join(args.basedir, 'images', image_filename(pidx, seed))

  batch_size = num_procs * args.per_batch_size
  num_chuncks = (len(all_captions) + batch_size - 1) // batch_size
  for seed in range(args.num_seeds):
    generator = torch.Generator('cuda').manual_seed(seed)
    for sidx in range(num_chuncks):
      sub_captions = all_captions[sidx * batch_size:(sidx + 1) * batch_size]
      with distributed_state.split_between_processes(sub_captions) as captions:
        # Continue if all images in the batch already generated.
        filenames = [img_fn(c_to_idx[caption], seed) for caption in captions]
        if all(os.path.exists(fn) for fn in filenames):
          continue

        # Generate the batch in one inference.
        img_results = pipe(captions, generator=generator).images
        for caption, img_result in zip(captions, img_results):
          filename = img_fn(c_to_idx[caption], seed)
          img_result.save(filename)

  distributed_state.wait_for_everyone()


def compute_rewards(args, captions, images, cqas):
  """Computes rewards and returns the final data dict."""
  images_paths = [os.path.join(args.basedir, 'images', img) for img in images]
  all_rewards = rewards.vqa_rewards(captions, images_paths, cqas)
  final_data_dicts = []
  for caption, image, reward in zip(captions, images, all_rewards):
    final_data_dicts.append({
      'image': image,
      'caption': caption,
      'reward': reward,
    })
  return final_data_dicts


def main():
  args = parse_args()

  cqa_file = args.cqa_file
  if not os.path.exists(cqa_file):
    cqa_file = os.path.join(args.basedir, args.cqa_file)
  if not os.path.exists(cqa_file):
    print(f'File does not exist: {args.cqa_file}')
    return

  with open(cqa_file, 'r') as f:
    cqas = json.load(f)
  all_captions = list(cqas.keys())
  c_to_idx = {caption: idx for idx, caption in enumerate(all_captions)}

  mkdir_p(os.path.join(args.basedir, 'images'))
  generate_images(args, all_captions, c_to_idx)

  images, captions = [], []
  for caption, seed in itertools.product(all_captions, range(args.num_seeds)):
    images.append(image_filename(c_to_idx[caption], seed))
    captions.append(caption)
  data_dicts = compute_rewards(args, captions, images, cqas)

  with open(os.path.join(args.basedir, 'sft_train_data.json'), 'w') as f:
    json.dump(data_dicts, f, indent=4)


if __name__ == '__main__':
  main()

"""Utils to compute various rewards."""

from PIL import Image

import clip
import hpsv2
import ImageReward as RM
from lavis.models import load_model_and_preprocess
import numpy as np
import torch
from tifascore import VQAModel
from transformers import AutoModel, AutoProcessor


clip_model, clip_preprocess = None, None  # CLIP.
blip_model, blip_vis_processor, blip_txt_processor = None, None, None
pickscore_model, pickscore_processor = None, None  # PickScore.
ir_model = None   # ImageReward.
vqa_model = None  # VQA with TIFA.

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def clip_score(captions, image_paths):
  global clip_model, clip_preprocess
  if not clip_model:
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  image_inputs = [
      clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
      for image_path in image_paths
  ]
  image_inputs = torch.concat(image_inputs, axis=0)
  text_inputs = clip.tokenize(captions).to(device)

  with torch.no_grad():
    # Cosine sim between the image and text features times 100.
    # Shape: [batch, batch].
    logits_per_image, _ = clip_model(image_inputs, text_inputs)
    # Only take the scores for pairwise image and text features.
    logits = torch.diagonal(logits_per_image, 0)

    # Cosine sim can also be computed as follows:
    #   image_features = clip_model.encode_image(image_inputs)
    #   text_features = clip_model.encode_text(text_inputs)
    #   image_features /= image_features.norm(dim=-1, keepdim=True)
    #   text_features /= text_features.norm(dim=-1, keepdim=True)
    #   similarity = image_features @ text_features.T
    #   similarity = torch.diagonal(similarity, 0)

  return (logits.cpu().numpy() / 100.0).tolist()


def blip_score(captions, image_paths):
  global blip_model, blip_vis_processor, blip_txt_processor
  if not blip_model:
    blips = load_model_and_preprocess(name='blip2_feature_extractor',
                                      model_type='pretrain', is_eval=True,
                                      device=device)
    blip_model, blip_vis_processor, blip_txt_processor = blips

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  image_inputs = [
      blip_vis_processor['eval'](
          Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
      for image_path in image_paths
  ]

  image_inputs = torch.concat(image_inputs, axis=0)
  text_inputs = [blip_txt_processor['eval'](caption) for caption in captions]
  sample = {'image': image_inputs, 'text_input': text_inputs}

  features_image = blip_model.extract_features(sample, mode='image')
  features_text = blip_model.extract_features(sample, mode='text')

  # Use low-dimensional projected features for similarity scoring.
  image_embeds_proj = features_image.image_embeds_proj  # [batch, 32, 256]
  text_embeds_proj = features_text.text_embeds_proj     # [batch, td, 256]
  similarity = image_embeds_proj @ text_embeds_proj[:, 0, :].unsqueeze(-1)
  similarity = torch.max(similarity, dim=1).values.squeeze(-1)
  return similarity.cpu().numpy().tolist()


def pick_score(captions, image_paths):
  global pickscore_model, pickscore_processor
  if not pickscore_model or not pickscore_processor:
    processor_name_or_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    model_pretrained_name_or_path = 'yuvalkirstain/PickScore_v1'
    pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path)
    pickscore_model.eval().to(device)

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  # Preprocess.
  images = [Image.open(image_path) for image_path in image_paths]
  image_inputs = pickscore_processor(
      images=images, padding=True, truncation=True, max_length=77,
      return_tensors='pt').to(device)
  text_inputs = pickscore_processor(
      text=captions, padding=True, truncation=True, max_length=77,
      return_tensors='pt').to(device)

  with torch.no_grad():
    # Embeddings: [batch_size, 1024].
    image_embs = pickscore_model.get_image_features(**image_inputs)
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    text_embs = pickscore_model.get_text_features(**text_inputs)
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

    # Pick score.
    embs_dot = text_embs @ image_embs.T
    # Only take the scores for pairwise image and text features.
    embs_dot = torch.diagonal(embs_dot, 0)
    scores = pickscore_model.logit_scale.exp() * embs_dot / 100.0

  return scores.cpu().numpy().tolist()


def image_reward(captions, image_paths):
  """Returns ImageReward rewards."""
  global ir_model
  if not ir_model:
    ir_model = RM.load('ImageReward-v1.0')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  rewards = []
  # scores = ir_model.score(captions, image_paths)
  # scores = scores if isinstance(scores, list) else [scores]
  # for i in range(len(image_paths)):
  #   per_image_scores = scores[len(captions)*i:len(captions)*(i+1)]
  #   rewards.append(per_image_scores[i])
  for caption, image_path in zip(captions, image_paths):
    rewards.append(ir_model.score(caption, [image_path]))
  return rewards


def hpsv2_reward(captions, image_paths):
  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  rewards = []
  for caption, image_path in zip(captions, image_paths):
    rewards.extend(hpsv2.score(image_path, caption))
  return rewards


def vqa_rewards(captions, image_paths, cqas):
  """Returns VQA-based rewards."""
  global vqa_model
  if not vqa_model:
    vqa_model = VQAModel('mplug-large')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  rewards = []
  for caption, image_path in zip(captions, image_paths):
    if caption not in cqas:
      print(f'Caption "{caption}" not found.')
      continue

    vqa_scores = []
    for question_answer_pair in cqas[caption]:
      choices = question_answer_pair['choices']
      question = question_answer_pair['question']
      vqa_answer = vqa_model.multiple_choice_vqa(image_path, question,
                                                 choices=choices)
      mc_answer = vqa_answer['multiple_choice_answer']
      score = int(mc_answer == question_answer_pair['answer'])
      vqa_scores.append(score)

    rewards.append(np.mean(vqa_scores))

  return rewards


if __name__ == '__main__':
  captions = [
      'an airplane to the left of an automobile',
      'an airplane to the left of an automobile',
      'a horse above an airplane',
      'a horse above an airplane',
      'a deer to the left of an automobile',
      'a deer to the left of an automobile',
  ]
  basedir = '/home/kykim/dev/t2i-eval/datasets/cifar10/spatial_relation/images'
  image_paths = [
      f'{basedir}/cifar10_0_0_0.jpg',
      f'{basedir}/cifar10_0_0_1.jpg',
      f'{basedir}/cifar10_54_1_1.jpg',
      f'{basedir}/cifar10_54_1_2.jpg',
      f'{basedir}/cifar10_92_0_0.jpg',
      f'{basedir}/cifar10_92_0_1.jpg',
  ]

  # print(f'CLIP: {clip_score(captions, image_paths)}')
  # for caption, image_path in zip(captions, image_paths):
  #   print(f'{clip_score(caption, image_path)}')

  # captions = ['a large fountain spewing water into the air']
  # image_paths = ['/home/kykim/dev/t2i-ft/sft/datasets/merlion.png']
  # print(f'BLIP: {blip_score(captions, image_paths)}')
  # for caption, image_path in zip(captions, image_paths):
  #   print(f'{blip_score(caption, image_path)}')

  # print(f'PickScore: {pick_score(captions, image_paths)}')
  # for caption, image_path in zip(captions, image_paths):
  #   print(f'{pick_score(caption, image_path)}')

  print(f'ImageReward: {image_reward(captions, image_paths)}')
  for caption, image_path in zip(captions, image_paths):
    print(f'{image_reward(caption, image_path)}')

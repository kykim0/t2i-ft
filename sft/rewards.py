
import numpy as np

from tifascore.vqa_models import VQAModel
from tqdm.auto import tqdm


def vqa_rewards(captions, images, cqas):
  """Returns VQA-based rewards."""
  vqa_model = VQAModel('mplug-large')

  rewards = []
  for caption, image in tqdm(zip(captions, images)):
    if caption not in cqas:
      print(f'Caption "{caption}" not found.')
      continue

    vqa_scores = []
    for question_answer_pair in cqas[caption]:
      choices = question_answer_pair['choices']
      question = question_answer_pair['question']
      vqa_answer = vqa_model.multiple_choice_vqa(image, question,
                                                 choices=choices)
      mc_answer = vqa_answer['multiple_choice_answer']
      score = int(mc_answer == question_answer_pair['answer'])
      vqa_scores.append(score)

    rewards.append(np.mean(vqa_scores))

  return rewards

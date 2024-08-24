import json
import os
import re
from collections import defaultdict
from pprint import pprint
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..dataset_zoo.config import DATASET_INFO, get_all_splits
from ..metrics.itm import compute_mllm_score_itg, is_match_with_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# set the max number of tiles in `max_num`
def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLModel:
    def __init__(self, model_name="OpenGVLab/InternVL-Chat-V1-5", device="cuda"):
        # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
        self.model = (
            AutoModel.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device = device

    def predict(self, text: str, image: Union[Image.Image, list]):
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        if isinstance(image, list):
            pixel_values1 = load_image(image[0], max_num=6).to(torch.bfloat16).cuda()
            pixel_values2 = load_image(image[1], max_num=6).to(torch.bfloat16).cuda()
            pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        else:
            pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()

        response, _ = self.model.chat(
            self.tokenizer, pixel_values, text, generation_config, history=None, return_history=True
        )
        return response

    def predict_batch(self, texts, images, max_new_tokens=512):
        return [self.predict(text, image) for text, image in zip(texts, images)]

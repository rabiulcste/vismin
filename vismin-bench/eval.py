import argparse
import os

import clip
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm

from metric import compute_contrastive_accuracy

# load vismin-bench dataset
DATA_PATH = ""  # set your HF dataset path
dataset_dict = load_from_disk(DATA_PATH)
test_dataset = dataset_dict["test"]
print(f"Dataset loaded successfully {len(test_dataset)} samples")

# load clip model
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="contrastive")
    args = parser.parse_args()

    predictions = []
    for sample in tqdm(test_dataset):
        text_0, text_1 = sample["text_0"], sample["text_1"]
        image_0, image_1 = sample["image_0"], sample["image_1"]
        category = sample["category"]

        text_tokens = clip.tokenize([text_0, text_1]).to("cuda")
        image_tokens = torch.stack([preprocess(image_0), preprocess(image_1)]).to(
            "cuda"
        )

        # Compute T0_I0, T0_I1, T1_I0, T1_I1 using the CLIP model
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            image_features = clip_model.encode_image(image_tokens)

            # Compute similarity scores (dot products) for all combinations
            T0_I0 = F.cosine_similarity(text_features[0], image_features[0], dim=0)
            T0_I1 = F.cosine_similarity(text_features[0], image_features[1], dim=0)
            T1_I0 = F.cosine_similarity(text_features[1], image_features[0], dim=0)
            T1_I1 = F.cosine_similarity(text_features[1], image_features[1], dim=0)

            predictions.append(
                {
                    "idx": sample["id"],
                    "T0_I0": T0_I0.item(),
                    "T0_I1": T0_I1.item(),
                    "T1_I0": T1_I0.item(),
                    "T1_I1": T1_I1.item(),
                    "category": category,
                }
            )

    metric_result = compute_contrastive_accuracy(predictions)
    print(f"Result on VisMin-Bench: \n")
    for category, result in metric_result.items():
        print(
            f"{category}: {result['text']:.4f}, {result['image']:.4f}, {result['group']:.4f}"
        )

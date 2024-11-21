import clip
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from metric import compute_contrastive_accuracy
import os
import torch.nn.functional as F
import argparse

# load vismin-bench dataset
CUSTOM_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(CUSTOM_PATH, "data")
dataset_dict = load_from_disk(DATA_PATH)
test_dataset = dataset_dict["test"]
print(test_dataset)

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
        image_tokens = torch.stack([preprocess(image_0), preprocess(image_1)]).to("cuda")

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
    ground_truth = pd.read_csv(os.path.join(CUSTOM_PATH, "solutions/sim_solution.csv"))
    ground_truth = ground_truth.to_dict(orient="records")
    assert len(predictions) == len(ground_truth), "Predictions and ground truth must have the same length"

    metric_result = compute_contrastive_accuracy(predictions, ground_truth)
    print(f"Result on VisMin-Bench: \n")
    for category, result in metric_result.items():
        print(f"{category}: {result['text']:.4f}, {result['image']:.4f}, {result['group']:.4f}")

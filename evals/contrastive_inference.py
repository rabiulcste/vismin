import argparse
from typing import Union

import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

from commons.logger import Logger
from commons.utils import set_random_seed

from .dataset_zoo.config import DATASET_INFO, get_all_splits
from .metrics.itm import (compute_contrastive_score_itg,
                          compute_contrastive_score_itg_by_tag,
                          compute_contrastive_score_standard)
from .vlm.model_factory import ModelFactory

logger = Logger.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed()


class VLMTaskPredictor:
    def __init__(self, dataset_name, model_name, pretrained=None):
        self.dataset_name = dataset_name
        self.model = ModelFactory(model_name, pretrained)
        self.task_info = self.get_task_info()
        self.task_type = self.task_info["name"]
        self.inference_function = self.task_info["function"]
        self.evaluation_metric = self.task_info["metric"]

    def get_task_info(self) -> dict:
        task_mapping = {
            "i2t_task": [
                "countbench",
                "vsr",
                "coco_spatial",
                "gqa_spatial",
                "spec_i2t",
                "whatsup_controlled",
                "whatsup_clevr",
                "sugarcrepe",
                "valse_v2",
            ],
            "t2i_task": ["spec_t2i", "imagecode"],
            "image_text_and_group_task": ["vismin", "winoground", "eqben", "mmvp"],
        }
        metric_mapping = {
            "i2t_task": compute_contrastive_score_standard,
            "t2i_task": compute_contrastive_score_standard,
            "image_text_and_group_task": compute_contrastive_score_itg,
        }
        for task_type, datasets in task_mapping.items():
            if self.dataset_name in datasets:
                return {
                    "name": task_type,
                    "function": getattr(self, f"infer_{task_type}"),
                    "metric": metric_mapping[task_type],
                }

        raise ValueError(f"Dataset name '{self.dataset_name}' not found in inference task lists.")

    def infer_i2t_task(self, dataset: torch.utils.data.Dataset, split_name: str):
        predictions = []
        
        for sample in tqdm(dataset, desc=split_name, total=len(dataset)):
            images, texts = sample["image"], sample["caption"]
            with torch.no_grad():
                logits_per_image = self.model.predict(texts, images)
                predicted_text_indices = torch.argmax(logits_per_image, dim=1)
                predictions.extend(predicted_text_indices.tolist())

        return predictions

    def infer_t2i_task(self, dataset: torch.utils.data.Dataset, split_name: str):
        predictions = []

        for sample in tqdm(dataset, desc=split_name, total=len(dataset)):
            images, texts = sample["image"], sample["caption"]
            with torch.no_grad():
                logits_per_image = self.model.predict(texts, images)
                logits_per_text = logits_per_image.t()
                predicted_image_indices = torch.argmax(logits_per_text, dim=1)
                predictions.extend(predicted_image_indices.tolist())

        return predictions

    def infer_image_text_and_group_task(self, dataset: torch.utils.data.Dataset, split_name):
        contrastive_scores = []
        for sample in tqdm(dataset, desc=split_name, total=len(dataset)):
            with torch.no_grad():
                # Combine images and captions into a list
                # pil image first
                images = [sample["image_0"], sample["image_1"]]
                texts = [sample["caption_0"], sample["caption_1"]]

                logits_per_image = self.model.predict(texts, images)

                # Extract the similarity scores using the logits
                sim_c0_i0 = logits_per_image[0, 0]
                sim_c0_i1 = logits_per_image[1, 0]
                sim_c1_i0 = logits_per_image[0, 1]
                sim_c1_i1 = logits_per_image[1, 1]

                curr_score = {
                    "c0_i0": sim_c0_i0.cpu().item(),
                    "c1_i0": sim_c1_i0.cpu().item(),
                    "c0_i1": sim_c0_i1.cpu().item(),
                    "c1_i1": sim_c1_i1.cpu().item(),
                    **({"tag": sample["tag"]} if "tag" in sample else {}),
                }
                contrastive_scores.append(curr_score)

        return contrastive_scores

    def run(self, overwrite_cache=False) -> Union[dict, float]:
        logger.info(f"Start evaluating {self.dataset_name} dataset")

        all_metrics = {}
        for split_name in get_all_splits(self.dataset_name):
            task_dataset = self.load_task_dataset(split_name)
            results = self.inference_function(task_dataset, split_name)

            metrics = self.evaluation_metric(results)
            all_metrics[split_name] = metrics

        for split_name, metrics in all_metrics.items():
            print(f"Metrics for {self.dataset_name}.{split_name}: {metrics}")

    def load_task_dataset(self, split_name):
        dataset_info = DATASET_INFO[self.dataset_name]
        img_root_dir = dataset_info[split_name]["img_root_dir"]
        annotation_file = dataset_info[split_name]["annotation_file"]
        print(f"split_name: {split_name}, img_root_dir: {img_root_dir}, annotation_file: {annotation_file}")
        task_dataset = dataset_info["cls"](
            img_root_dir=img_root_dir, annotation_file=annotation_file, model_type="clip"
        )
        return task_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="vismin",
        choices=[
            "eqben",
            "eqben_all",
            "winoground",
            "valse_v2",
            "sugarcrepe",
            "coco_negatives",
            "countbench",
            "vsr",
            "vismin",
            "mmvp",
            "coco_spatial",
            "gqa_spatial",
            "spec_i2t",
            "spec_t2i",
            "imagecode",
            "whatsup_controlled",
            "whatsup_clevr",
        ],
        help="Specify the evaluation dataset",
    )
    parser.add_argument("--model_name", default="clip_vit_b32", help="clip_rn50, clip_vit_b32")
    parser.add_argument(
        "--pretrained",
        default="openai",
        # choices=["openai", "open_clip", "google"],
        help="Specify the pretrained model developer",
    )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cache")
    args = parser.parse_args()

    # model_name = args.pretrained if "checkpoints" in args.pretrained else args.model_name
    # pretrained = args.pretrained
    print(f"Model name: {args.model_name}, Pretrained: {args.pretrained}")
    task_predictor = VLMTaskPredictor(args.dataset, args.model_name, args.pretrained)
    task_predictor.run(args.overwrite_cache)

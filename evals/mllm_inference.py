import argparse
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from typing import Any, List, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from commons.logger import Logger
from commons.utils import set_random_seed

from .dataset_zoo.config import DATASET_INFO, get_all_splits
from .metrics.itm import (compute_mllm_score_itg, compute_mllm_score_standard,
                          is_match_with_label)
from .vlm.model_factory import ModelFactory

set_random_seed()
logger = Logger.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


class VLMTaskPredictor:
    def __init__(self, dataset_name, model_name):
        logger.info(f"Dataset: {dataset_name}, Model: {model_name}")
        self.dataset_name = dataset_name
        self.model = ModelFactory(model_name)
        self.task_info = self.get_task_info()
        self.task_type = self.task_info["name"]
        self.inference_function = self.task_info["function"]
        self.evaluation_metric = self.task_info["metric"]

    def get_task_info(self) -> dict:
        task_mapping = {
            "standard_task": [
                "imagecode",
                "spec_t2i",
                "vsr",
                "countbench",
                "valse_v2",
                "sugarcrepe",
                "coco_spatial",
                "gqa_spatial",
                "spec_i2t",
                "spec_t2i",
                "whatsup_controlled",
                "whatsup_clevr",
            ],
            "image_text_and_group_task": ["vismin", "winoground", "eqben", "mmvp"],
        }
        metric_mapping = {
            "standard_task": compute_mllm_score_standard,
            "image_text_and_group_task": compute_mllm_score_itg,
        }
        for task_type, datasets in task_mapping.items():
            if self.dataset_name in datasets:
                return {
                    "name": task_type,
                    "function": getattr(self, f"infer_{task_type}"),
                    "metric": metric_mapping[task_type],
                }

        raise ValueError(f"Dataset name '{self.dataset_name}' not found in inference task lists.")

    def infer_standard_task(self, batch: list):
        images, texts, labels = batch["image"], batch["text"], batch["label"]
        predictions = self.model.predict_batch(texts, images, max_new_tokens=128)
        return self.generate_standard_results(images, texts, predictions, labels)

    def infer_image_text_and_group_task(self, batch: list):
        task_results = {}
        batch_items, batch_ids = batch["items"], batch["id"]
        for sample_id, items in zip(batch_ids, batch_items):
            texts = [item["text"] for item in items]
            images = [item["image"] for item in items]
            predictions = self.model.predict_batch(texts, images, max_new_tokens=128)
            task_results[sample_id] = self.generate_grouped_task_results(items, predictions)
        return task_results

    def generate_standard_results(self, images, texts, predictions, labels):
        results = []
        for image, text, prediction, label in zip(images, texts, predictions, labels):
            match = is_match_with_label(prediction, label)
            logger.debug(f"Task: {text} => {prediction} # GT = {label} => Match = {match}")
            results.append({"image": image, "text": text, "prediction": prediction, "label": label, "match": match})
        return results

    def generate_grouped_task_results(self, items, predictions):
        task_results = defaultdict(list)
        for item, prediction in zip(items, predictions):
            task_type, label = item["type"], item["label"]
            match = is_match_with_label(prediction, label)
            logger.debug(f"Task: {item} => {prediction} # GT = {label} => Match = {match}")
            task_results[task_type].append({"prompt": item["text"], "prediction": prediction, "match": match})
        return task_results

    def run(self, overwrite_cache=False):
        all_metrics = {}

        for split_name in get_all_splits(self.dataset_name):
            cache_dir = self.setup_cache_directory(split_name, overwrite_cache)
            task_dataset = self.load_task_dataset(split_name)
            dataloader = DataLoader(task_dataset, batch_size=8, shuffle=False, collate_fn=task_dataset.collate)

            for batch in tqdm(dataloader, desc=split_name):
                batch_results = self.inference_function(batch)
                self.update_cache(cache_dir, batch, batch_results)

            cached_results = self.load_cached_results(cache_dir)
            metrics = self.evaluation_metric(cached_results)
            all_metrics[split_name] = metrics

        for split_name, metrics in all_metrics.items():
            print(f"Metrics for {self.dataset_name}.{split_name}: {metrics}")

    def load_task_dataset(self, split_name):
        dataset_info = DATASET_INFO[self.dataset_name]
        img_root_dir = dataset_info[split_name]["img_root_dir"]
        annotation_file = dataset_info[split_name]["annotation_file"]
        task_dataset = dataset_info["cls"](
            img_root_dir=img_root_dir, annotation_file=annotation_file, model_type="mllm"
        )
        return task_dataset

    def setup_cache_directory(self, split_name, overwrite_cache: bool) -> str:
        output_directory = os.path.join("evals", "outputs", "cache", self.dataset_name, split_name)
        if overwrite_cache and os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.makedirs(output_directory, exist_ok=True)
        return output_directory

    def update_cache(self, cache_directory: str, batch: List[dict], results: List[dict]):
        cache_file_path = os.path.join(cache_directory, "cache.json")
        cache = {}
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                cache = json.load(file)
        for sample_id in batch["id"]:
            cache[str(sample_id)] = results[sample_id]
        with open(cache_file_path, "w") as file:
            json.dump(cache, file, indent=2)

    def load_cached_results(self, cache_directory: str) -> dict:
        cache_file_path = os.path.join(cache_directory, "cache.json")
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                return json.load(file)
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and print results for image-text matching tasks.")
    parser.add_argument("--dataset", required=True, help="Dataset to use for evaluation")
    parser.add_argument("--model_name", required=True, help="Model name for predictions")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cache files")
    args = parser.parse_args()

    task_predictor = VLMTaskPredictor(args.dataset, args.model_name)
    task_predictor.run(args.overwrite_cache)

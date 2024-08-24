import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict
from typing import Union

import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ..utils.constants import (SYNTH_DIFFUSE_DATA_DIR, VALID_CATEGORY_NAMES,
                               VIM_DATA_DIR)
from ..utils.helpers import (crop_image_region_from_bbox_list, format_caption,
                             get_coco_path_by_image_id, get_coco_tags,
                             load_json_data, load_vsr_captions, pluralize)
from ..utils.logger import Logger
from ..utils.spatial_relation_utils import (get_xyxy_from_xywh,
                                            swap_phrases_in_spatial_prompt)

VALID_SPATIAL_DIRECTIONS = ["left", "right", "top", "bottom", "below", "above", "under"]

logger = Logger.get_logger(__name__)
# Setting a style for the plot
sns.set_style("whitegrid")


def extract_spatial_direction_from_caption(caption: str):
    words = caption.split()
    positions = {direction: words.index(direction) for direction in VALID_SPATIAL_DIRECTIONS if direction in words}
    if not positions:
        return None
    return min(positions, key=positions.get)


def plot_thresholded_subcategory_distribution(subcategory_distribution, category_name: str, split: str):
    sorted_category = dict(sorted(subcategory_distribution.items(), key=lambda item: item[1], reverse=True))
    # print(f"Category: {category_name}, Subcategory Distribution: {sorted_category}")
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    ax = sns.barplot(x=list(sorted_category.keys()), y=list(sorted_category.values()), palette="Blues_d")

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title(f"Subcategory Distribution for {category_name}", fontsize=16)

    plt.tight_layout()

    fname = f"diffused_generator/output/data_analysis/{category_name}_{split}_subcategory_distribution.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, bbox_inches="tight", dpi=600)
    plt.close()

    logger.info(f"Subcategory distribution plot for {category_name} saved to {fname}")


def compute_and_plot_thresholded_subcategory_distribution(data: list, split: str, min_count: int = 20):
    category_subcategory_distribution = defaultdict(lambda: defaultdict(int))
    for sample in data:
        for info in sample["negatives"]:
            category = info["category"]
            sub_category = info["edited_phrase"] if category == "object" else info["sub_category"]
            category_subcategory_distribution[category][sub_category] += 1

    for category, subcategory_dist in category_subcategory_distribution.items():
        thresholded_distribution = {
            sub_category: count for sub_category, count in subcategory_dist.items() if count >= min_count
        }
        logger.debug(f"Category dist. stat: {thresholded_distribution}")
        plot_thresholded_subcategory_distribution(thresholded_distribution, category, split)


def extract_category_and_subcategory(category_str: str):
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", category_str)
    if match:
        category, sub_category = match.groups()
    else:
        category = category_str.split("(")[0].strip()
        sub_category = "others"

    if category not in VALID_CATEGORY_NAMES:
        category = "others"
    return category, sub_category


def save_to_jsonl(train_set, fname):
    with open(fname, "w") as f:
        for item in train_set:
            json.dump(item, f)
            f.write("\n")
    logger.info(f"Saved to {fname}")


def export_data_by_category_to_jsonl(dataset, output_dir):
    for category_name, train_data in dataset.items():
        if category_name not in VALID_CATEGORY_NAMES:
            continue
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{category_name}.jsonl")
        save_to_jsonl(train_data, fname)


class DataPreparEngine:
    def __init__(self):
        print("Data Preparation Engine Initialized")

    def prepare_data(self, args):
        split_str = f"{args.split}{'ing' if args.split == 'train' else ''}"
        logger.info("=" * 50)
        logger.info(f"🔥 Starting Preparation of {split_str.capitalize()} Data 🔥")
        logger.info("=" * 50)
        print("\n")
        # input data paths
        coco_data_dir_path = os.path.join(VIM_DATA_DIR, f"coco_sdxl_edited_{args.split}")
        vsr_data_dir_path = os.path.join(VIM_DATA_DIR, f"vsr_sdxl_edited_{args.split}")
        coco_counting_data_dir_path = os.path.join(VIM_DATA_DIR, f"coco_sdxl_removed_{args.split}")
        spatial_syn_data_dir_path = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_relation_{args.split}")
        counting_syn_data_dir_path = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_counting_{args.split}")

        # output data paths
        default_output_dir_path = os.path.join(VIM_DATA_DIR, "annotations", f"{split_str}_data")
        if args.enable_mturk:
            default_output_dir_path += ".mturk.may12"

        # object, attribute, counting
        data_coco = self.prepare_data_coco(
            coco_data_dir_path, coco_counting_data_dir_path, args.split, args.enable_region_crop
        )
        data_vsr = self.prepare_data_coco(vsr_data_dir_path, "unknown", args.split, args.enable_region_crop)
        data_synthetic_spatial = self.prepare_data_layout_based(
            spatial_syn_data_dir_path, "relation", args.split, args.enable_region_crop
        )
        data_synthetic_counting = self.prepare_data_layout_based(
            counting_syn_data_dir_path, "counting", args.split, args.enable_region_crop
        )
        all_data = data_coco + data_vsr + data_synthetic_spatial + data_synthetic_counting
        negative_cnt = sum([len(sample["negatives"]) for sample in all_data])
        logger.info(f"Number of total {split_str} samples => positive: {len(all_data)}, negative: {negative_cnt}")

        os.makedirs(default_output_dir_path, exist_ok=True)
        all_data_fpath = os.path.join(default_output_dir_path, f"{args.split}.json")
        with open(all_data_fpath, "w") as f:
            json.dump(all_data, f, indent=2)
        logger.info(f"Saved {split_str} set to {all_data_fpath}")
        time.sleep(10)

        if args.split == "validation" and args.enable_mturk:
            data_by_category = self.sample_data_by_subcategory(all_data)
        else:
            data_by_category = self.save_data_by_category(all_data)

        export_data_by_category_to_jsonl(data_by_category, default_output_dir_path)
        self.export_data_by_category_to_csv(all_data, args.split)

        logger.info(f"📊 Plotting {split_str} data")
        compute_and_plot_thresholded_subcategory_distribution(all_data, args.split)

        logger.info(f"✅ {split_str.capitalize()} Data Preparation Complete!")
        print("\n\n")

    def export_generative_data(self, data, split):
        json_path = os.path.join(VIM_DATA_DIR, "annotations", "generative", split, "visdiff_ours.json")
        visdiff_anns = load_json_data(json_path)

        visdiff_data = []
        for sample in data:
            image_id = sample["image_id"]
            sample["caption"] = format_caption(sample["caption"])
            for info in sample["negatives"]:
                info["edited_caption"] = format_caption(info["edited_caption"])
                visual_difference = visdiff_anns.get(image_id, {}).get(info.get("edit_id"), None)
                if visual_difference is None:
                    continue
                visdiff_data.append(
                    {
                        "image_path": sample["image_path"],
                        "edited_image_path": info["edited_image_path"],
                        "visdiff": visual_difference,
                        "edit_id": info.get("edit_id"),
                    }
                )

        out_fpath = os.path.join(VIM_DATA_DIR, "annotations", "generative", split, f"{split}_ours.csv")
        with open(out_fpath, "w", newline="") as f:
            fieldnames = ["image_path", "edited_image_path", "visdiff", "edit_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for item in visdiff_data:
                writer.writerow(item)

        logger.info(f"Saved generative data to {out_fpath}")

    def export_data_by_category_to_csv(self, training_data, split):
        training_data_by_category = defaultdict(list)
        for sample in training_data:
            sample["caption"] = format_caption(sample["caption"])
            source_set = set()
            for info in sample["negatives"]:
                category = info["category"]
                info["edited_caption"] = format_caption(info["edited_caption"])

                if category not in source_set:
                    training_data_by_category[category].append(
                        {
                            "image_path": sample["image_path"],
                            "caption": sample["caption"],
                        }
                    )
                    source_set.add(category)

                training_data_by_category[category].append(
                    {
                        "image_path": info["edited_image_path"],
                        "caption": info["edited_caption"],
                    }
                )
        output_dir = os.path.join(VIM_DATA_DIR, "annotations", "contrastive", split)
        os.makedirs(output_dir, exist_ok=True)

        # save category wise data to csv, keep headings as image_path, caption
        for category, samples in training_data_by_category.items():
            if category not in VALID_CATEGORY_NAMES:
                continue
            fpath = os.path.join(output_dir, f"{category}.csv")
            with open(fpath, "w") as f:
                f.write("image_path\tcaption\n")
                for sample in samples:
                    f.write(f"{sample['image_path']}\t{sample['caption']}\n")
            logger.info(f"Saved contrastive data to {fpath}")

    def save_data_by_category(self, training_data):
        positive_counts = defaultdict(int)
        training_data_by_category = defaultdict(list)

        for sample in training_data:
            image_source = sample["source"]
            positive_counts[image_source] += 1
            for info in sample["negatives"]:
                category = info["category"]
                sample["caption"] = format_caption(sample["caption"])
                info["edited_caption"] = format_caption(info["edited_caption"])
                training_data_by_category[category].append(
                    {
                        "image_id": sample["image_id"],
                        "image_path": sample["image_path"],
                        "caption": sample["caption"],
                        "edited_image_path": info["edited_image_path"],
                        "edited_caption": info["edited_caption"],
                        "input_phrase": info.get("input_phrase"),
                        "edited_phrase": info.get("edited_phrase"),
                        "sub_category": info.get("sub_category"),
                        "edit_id": info.get("edit_id"),
                        "region_crop_bool": info.get("region_crop_bool", False),
                    }
                )

        formatted_counts = "\n".join([f"- {category}: {count} samples" for category, count in positive_counts.items()])
        logger.info(f"Sample counts by category (positive):\n{formatted_counts}")

        sample_count_by_category = {
            category: len(samples) for category, samples in training_data_by_category.items() if len(samples) > 20
        }
        formatted_counts = "\n".join(
            [f"- {category}: {count} samples" for category, count in sample_count_by_category.items()]
        )
        logger.info(f"Sample counts by category (negative):\n{formatted_counts}")

        return training_data_by_category

    def sample_data_by_subcategory(self, training_data):
        training_data_by_category = defaultdict(lambda: defaultdict(list))
        for sample in training_data:
            random.shuffle(sample["negatives"])
            for info in sample["negatives"]:
                category = info["category"]
                sub_category = "spatially-aware" if sample["source"] == "vsr" else info["sub_category"]
                sample["caption"] = format_caption(sample["caption"])
                info["edited_caption"] = format_caption(info["edited_caption"])

                if sub_category == "No Attribute":
                    logger.debug(f"Category: {category}, Subcategory: {sub_category} is not valid")
                    continue
                training_data_by_category[category][sub_category].append(
                    {
                        "image_id": sample["image_id"],
                        "image_path": sample["image_path"],
                        "caption": sample["caption"],
                        "edited_image_path": info["edited_image_path"],
                        "edited_caption": info["edited_caption"],
                        "input_phrase": info.get("input_phrase"),
                        "edited_phrase": info.get("edited_phrase"),
                        "sub_category": sub_category,
                        "edit_id": info.get("edit_id"),
                        "region_crop_bool": info.get("region_crop_bool", False),
                    }
                )

        MAX_SAMPLES_PER_CATEGORY = {
            "object": 3000,
            "attribute": 3000,
            "relation": 10450,
            "counting": 13850,
        }
        print(f"MAX_SAMPLES_PER_CATEGORY (THRESHOLD): {MAX_SAMPLES_PER_CATEGORY}")
        adjusted_training_data = {}
        for category, sub_categories in training_data_by_category.items():
            total_samples = sum(len(samples) for samples in sub_categories.values())
            num_sub_categories = max(1, len(sub_categories))
            base_allocation = max(1, MAX_SAMPLES_PER_CATEGORY[category] // num_sub_categories)
            print(f"===== category: {category} ======")
            print(f"total samples: {total_samples}, per sub_category fair allocation: {base_allocation}")

            adjusted_sub_category_data = {}
            for sub_category, samples in sub_categories.items():
                random.shuffle(samples)
                adjusted_samples = samples[:base_allocation]
                adjusted_sub_category_data[sub_category] = adjusted_samples

            adjusted_training_data[category] = adjusted_sub_category_data

        sample_count_by_category = {
            category: {sub_category: len(data) for sub_category, data in sub_categories.items()}
            for category, sub_categories in adjusted_training_data.items()
        }

        # sort the subcategories by count
        for category, sub_categories in sample_count_by_category.items():
            sample_count_by_category[category] = dict(
                sorted(sub_categories.items(), key=lambda item: item[1], reverse=True)
            )

        formatted_counts = "\n".join(
            [
                f"- {category}: {sub_category} has {count} samples"
                for category, sub_categories in sample_count_by_category.items()
                for sub_category, count in sub_categories.items()
            ]
        )

        logger.info(f"Sample counts by category:\n{formatted_counts}")

        final_data = {}
        for category, sub_categories in adjusted_training_data.items():
            curr_data = []
            for sub_category, samples in sub_categories.items():
                curr_data.extend(samples)
            final_data[category] = curr_data[: MAX_SAMPLES_PER_CATEGORY[category]]

        return final_data

    def load_llm_edits_filtering_data(self, dataset, split):
        data = {}
        if dataset == "coco" and split == "train":
            dir_path = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR,
                "prompt_resources",
                f"llm_edits_{dataset}",
                "mistralaimixtral8x7binstructv0.1",
                f"edit_instruction_filtered_{split}_chunked",
            )
            for chunk_json_fname in os.listdir(dir_path):
                chunk_fpath = os.path.join(dir_path, chunk_json_fname)
                with open(chunk_fpath, "r") as f:
                    curr_data = json.load(f)
                data.update(curr_data)
        else:
            json_fpath = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR,
                "prompt_resources",
                f"llm_edits_{dataset}",
                "mistralaimixtral8x7binstructv0.1",
                f"edit_instruction_filtered_{split}.json",
            )
            with open(json_fpath, "r") as f:
                data = json.load(f)

        return data

    def load_llm_tagged_subcategory_data(self, dataset: str, split: str, category: str):
        fname = (
            f"{category}_tagging_{split}_openai.json" if split == "validation" else f"{category}_tagging_{split}.json"
        )
        json_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset}",
            "mistralaimixtral8x7binstructv0.1",
            fname,
        )
        if not os.path.exists(json_fpath):
            logger.warning(f"File not found: {json_fpath}")
            return {}

        logger.info(f"Loading tagged {category} data from {json_fpath}")
        with open(json_fpath, "r") as f:
            data = json.load(f)

        return data

    def load_llm_refine_caption_counting_data(self, split):
        json_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            "llm_edits_coco",
            "mistralaimixtral8x7binstructv0.1",
            f"counting_caption_refined_{split}.json",
        )
        if not os.path.exists(json_fpath):
            return {}
        with open(json_fpath, "r") as f:
            data = json.load(f)

        return data

    def prepare_data_coco(self, dir_path: str, dir_path_count: str, split: str, enable_region_crop: bool = False):
        coco_split = "val" if split == "validation" else split
        imageid_tags, _, _, categories_id_to_supercategory = get_coco_tags(coco_split)

        TIFA_THRESHOLD = 0.95 if split == "train" else 0.98
        if not os.path.exists(dir_path) and not os.path.exists(dir_path_count):
            return []

        image_dirs = os.listdir(dir_path) if os.path.exists(dir_path) else []
        logger.info(f"Path: {dir_path}")

        dataset = "vsr" if "vsr" in dir_path else "coco"
        llm_attribute_tagging_data = self.load_llm_tagged_subcategory_data(dataset, split, "attribute")
        llm_object_tagging_data = self.load_llm_tagged_subcategory_data(dataset, split, "object")
        llm_edits_filtering_data = self.load_llm_edits_filtering_data(dataset, split)
        llm_refined_caption_counting_data = self.load_llm_refine_caption_counting_data(split)
        # intersection of the image directories with the llm edits filtering data
        image_dirs = list(set(image_dirs) & set(llm_edits_filtering_data.keys()))

        image_dirs_cnt = os.listdir(dir_path_count) if os.path.exists(dir_path_count) else []
        logger.info(f"Number of image directories (general): {len(image_dirs)} and (counting): {len(image_dirs_cnt)}")

        # first intersection of the image directories with the llm edits filtering data and then union with the counting ones
        image_dirs = list(set(image_dirs) | set(image_dirs_cnt))
        logger.info(f"Number of image directories after filtering: {len(image_dirs)}")
        training_data = []
        total_negative_samples_objattr, total_negative_samples_count = 0, 0
        total_invalid_image_dir, total_tifa_failure, total_tifa = 0, 0, 0
        object_cntr_limit_dict = {"1": 0, "2": 15} if split == "validation" else {"1": 3000, "2": 3000}

        # specific to vsr dataset
        vsr_anns = load_vsr_captions()
        vsr_anns = {str(ann["image_id"]): ann for ann in vsr_anns}
        for image_id in tqdm(image_dirs, desc="Image Directories"):
            # handling object, attribute, action categories
            json_file_path = os.path.join(dir_path, image_id, "annotations.json")
            objattr_negatives, region_crop_negatives = [], []
            region_crops_info = {}
            if os.path.exists(json_file_path):
                curr_dict = load_json_data(json_file_path)
                if curr_dict is None:
                    # shutil.rmtree(os.path.join(dir_path, image_id))
                    logger.warning(f"Invalid json file: {json_file_path}")
                    continue

                # if "input_image_path" not in curr_dict and dataset == "vsr":
                #     image_path = vsr_anns[image_id]["image_path"]
                # if vsr_anns.get(image_id, {}).get("image_path") is not None:
                #     curr_dict["input_image_path"] = image_path
                #     with open(json_file_path, "w") as f:
                #         json.dump(curr_dict, f, indent=2)
                if dataset == "vsr":
                    image_path = curr_dict["input_image_path"]
                else:
                    image_path = get_coco_path_by_image_id(split=coco_split, image_id=image_id)

                if "train" in image_path and split == "validation":
                    continue

                if dataset == "coco":
                    curr_cat_ids = list(set([data["category_id"] for data in imageid_tags[int(image_id)]]))
                    super_category_names = list(set([categories_id_to_supercategory.get(iid) for iid in curr_cat_ids]))
                    super_category_names_str = random.choice(super_category_names) if super_category_names else "Others"

                input_caption_text = curr_dict.get("input_caption")
                for info in curr_dict["annotations"]:
                    edited_image_path = list(info["scores"].keys())[0]  # we generate only one image
                    edited_caption_text = info.get("edited_caption")
                    if not any(word in info["category"] for word in ["object", "attribute"]):
                        logger.warning(f"Category not in object, attribute: {info['category']}")
                        continue
                    # use llm edits filter to filter out the ones that doesn't meet the criteria
                    llm_filter_curr = (
                        llm_edits_filtering_data.get(str(image_id), {}).get(info["edit_id"], {}).get("reject")
                    )
                    if llm_filter_curr != "NO":
                        continue
                    category, sub_category = extract_category_and_subcategory(info["category"])

                    scores = info["scores"]
                    tifa_score = scores[edited_image_path].get("tifa_score")
                    region_tifa_score = scores[edited_image_path].get("region_tifa_score")

                    if tifa_score is not None and region_tifa_score is not None:
                        total_tifa += 1

                    if tifa_score is None or region_tifa_score is None:
                        continue
                    if tifa_score < TIFA_THRESHOLD or region_tifa_score < TIFA_THRESHOLD:
                        total_tifa_failure += 1
                        continue
                    if split == "validation":
                        if tifa_score is None or region_tifa_score is None:
                            continue
                        if tifa_score < TIFA_THRESHOLD or region_tifa_score < TIFA_THRESHOLD:
                            total_tifa_failure += 1
                            continue
                    else:
                        if tifa_score is not None and tifa_score < TIFA_THRESHOLD or region_tifa_score < TIFA_THRESHOLD:
                            total_tifa_failure += 1
                            continue
                        elif region_tifa_score < TIFA_THRESHOLD:
                            continue
                    if not os.path.exists(edited_image_path):
                        logger.warning(f"Edited image path not found: {edited_image_path}")

                    if category == "attribute":
                        sub_category = llm_attribute_tagging_data.get(str(image_id), {}).get(info["edit_id"], "Others")
                    elif category == "object" and dataset == "coco":
                        sub_category = llm_object_tagging_data.get(str(image_id), {}).get(info["edit_id"], "Others")

                    objattr_negatives.append(
                        {
                            "edited_image_path": edited_image_path,
                            "edited_caption": edited_caption_text,
                            "input_phrase": info.get("input_phrase"),
                            "edited_phrase": info.get("edited_phrase"),
                            "category": category,
                            "sub_category": sub_category,
                            "edit_id": info["edit_id"],
                        }
                    )
                    # add region crop image
                    if split == "train" and enable_region_crop:
                        region_crops_info[info["input_phrase"]] = {
                            "id": image_id,
                            "bounding_boxes": info.get("bounding_boxes"),
                            "image_path": image_path,
                            "caption": info.get("input_phrase"),
                            "category": category,
                            "sub_category": sub_category,
                        }
                        region_crops_info[info["edited_phrase"]] = {
                            "id": info["edit_id"],
                            "bounding_boxes": info.get("bounding_boxes"),
                            "image_path": edited_image_path,
                            "caption": info.get("edited_phrase"),
                            "category": category,
                            "sub_category": sub_category,
                        }

                region_crop_negatives = self.process_image_crops_and_create_negative_dict(region_crops_info)
                total_negative_samples_objattr += len(objattr_negatives)
                total_negative_samples_objattr += len(region_crop_negatives)

            # handling count category
            json_file_path = os.path.join(dir_path_count, image_id, "annotations.json")
            count_negatives = []
            if os.path.exists(json_file_path):
                annotations = load_json_data(json_file_path)
                if annotations is None:
                    continue

                if annotations["image_edit_source"] == "open_ended":
                    continue
                image_path = annotations["image_path"]
                input_caption_text = annotations.get("input_caption")

                unique_caption = set()
                for info in annotations["edit_instructions"]:
                    if info.get("new_image_path") is None or info.get("tifa_score") != 1:
                        continue
                    if info["new_caption"] in unique_caption:
                        continue
                    if str(info["original_object_count"]) in list(object_cntr_limit_dict.keys()):
                        if object_cntr_limit_dict[str(info["original_object_count"])] <= 0:
                            continue
                        object_cntr_limit_dict[str(info["original_object_count"])] -= 1  # decrease the limit

                    unique_caption.add(info["new_caption"])
                    input_object_plural = (
                        pluralize(info["object_name"]) if info["original_object_count"] > 1 else info["object_name"]
                    )
                    edited_object_plural = (
                        pluralize(info["object_name"]) if info["original_object_count"] - 1 > 1 else info["object_name"]
                    )
                    input_phrase = f"{info['original_object_count']} {input_object_plural}"
                    edited_phrase = f"{info['original_object_count'] - 1} {edited_object_plural}"
                    count_negatives.append(
                        {
                            "edited_image_path": info["new_image_path"],
                            "edited_caption": llm_refined_caption_counting_data.get(image_id) or info["new_caption"],
                            "input_phrase": input_phrase,
                            "edited_phrase": edited_phrase,
                            "category": "counting",
                            "sub_category": str(info["original_object_count"]),
                            "edit_id": info["id"],
                        }
                    )
                total_negative_samples_count += len(count_negatives)

            if count_negatives or objattr_negatives:
                curr_neg_data = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption": input_caption_text,
                    "source": dataset,
                    "negatives": objattr_negatives + count_negatives,
                }
                if region_crop_negatives:
                    curr_neg_data["region_negatives"] = region_crop_negatives
                training_data.append(curr_neg_data)
            else:
                total_invalid_image_dir += 1

        logger.info(
            f"Number of samples => positive: {len(training_data)} negative (objattr): {total_negative_samples_objattr} negative (counting): {total_negative_samples_count}"
        )
        logger.info(
            f"Number of tifa failures: {total_tifa_failure} out of {total_tifa}, percentage: {total_tifa_failure/total_tifa* 100:.2f}"
        )
        logger.info(f"No valid images found directory count => {total_invalid_image_dir}")
        return training_data

    def prepare_data_layout_based(self, data_dir_path, category, split, enable_region_crop=False, enable_sbert=False):
        TIFA_THRESHOLD_SPATIAL = 0.9 if split == "train" else 0.95
        TIFA_THRESHOLD_COUNTING = 0.9
        image_dirs = os.listdir(data_dir_path)
        logger.info(f"Path: {data_dir_path}")
        logger.info(f"Number of image directories: {len(image_dirs)}")

        training_data = []
        total_negative_samples, total_invalid_image_dir = 0, 0
        model = SentenceTransformer("paraphrase-distilroberta-base-v1")  # Load the pre-trained SBERT model
        non_paired_samples = []
        for image_id in tqdm(image_dirs, desc="Image Directories"):
            json_file_path = os.path.join(data_dir_path, image_id, "annotations.json")
            data = load_json_data(json_file_path)
            if data is None:
                bad_dir_path = os.path.join(data_dir_path, image_id)
                if "lock" not in bad_dir_path:
                    shutil.rmtree(bad_dir_path)
                continue
            annotations = data["annotations"]
            scores = annotations.get("scores", {})
            image_path = annotations["generated_images"][0]
            negatives, region_crop_negatives = [], []
            if category == "relation":
                edited_image_path, edited_prompt = self.get_image_paths_and_edited_prompt_for_relation(
                    annotations, category, image_path, TIFA_THRESHOLD_SPATIAL
                )
                if edited_prompt is not None and edited_image_path is not None:
                    if Levenshtein.distance(sorted(annotations["prompt"]), sorted(edited_prompt)) > 5:
                        continue
                    negatives.append(
                        {
                            "edited_image_path": edited_image_path,
                            "edited_caption": edited_prompt,
                            "category": category,
                            "sub_category": extract_spatial_direction_from_caption(edited_prompt),
                            "edit_id": image_id,
                            "input_phrase": annotations["scores"][edited_image_path].get("input_phrase"),
                            "edited_phrase": annotations["scores"][edited_image_path].get("edited_phrase"),
                        }
                    )
                    inp_image = Image.open(image_path).convert("RGB")
                    if split == "train" and enable_region_crop:
                        for object_name, bbox in annotations["bounding_boxes"]:
                            cropped_image = inp_image.crop(get_xyxy_from_xywh(bbox))
                            edit_crop_id = f"{image_id}_{object_name}"
                            edit_crop_fname = re.sub(r"\s+|/", "", f"{edit_crop_id}.png")
                            img_crop_fpath = os.path.join(VIM_DATA_DIR, "region_crop_images", edit_crop_fname)
                            os.makedirs(os.path.dirname(img_crop_fpath), exist_ok=True)
                            if not os.path.exists(img_crop_fpath):
                                cropped_image.save(img_crop_fpath)
                            region_crop_negatives.append(
                                {
                                    "edited_image_path": img_crop_fpath,
                                    "edited_caption": object_name,
                                    "category": category,
                                    "edit_id": edit_crop_id,
                                    "input_phrase": annotations["prompt"],
                                    "edited_phrase": object_name,
                                    "region_crop_bool": True,
                                }
                            )

                elif enable_sbert:
                    if scores.get(image_path, {}).get("tifa_score") != 1:
                        continue  # skip
                    non_paired_samples.append(
                        {
                            "image_path": random.choice(annotations["generated_images"]),
                            "caption": annotations["prompt"],
                            "category": annotations["category"],
                        }
                    )

            elif category == "counting":
                edit_instructions = annotations.get("edit_instructions", [])
                for info in edit_instructions:
                    neg_image_path = os.path.join(data_dir_path, image_id, f"{info['edit_id']}.png")
                    if scores.get(neg_image_path, {}).get("tifa_score") < TIFA_THRESHOLD_COUNTING:
                        continue
                    # print(f"{scores.get(neg_image_path, {}).get('llm_qa_verify')}")
                    if info.get("edited_caption") is None:
                        continue
                    if info["edited_caption"].startswith("A group of one"):
                        info["edited_caption"] = info["edited_caption"].replace("A group of one", "One")
                        # print(f"Edited caption: {info['edited_caption']}")
                        continue

                    if info["edited_caption"].startswith("A group of no"):
                        info["edited_caption"] = info["edited_caption"].replace("A group of no", "No")
                        # print(f"Edited caption: {info['edited_caption']}")
                        continue

                    if Levenshtein.distance(sorted(annotations["prompt"]), sorted(info["edited_caption"])) > 5:
                        continue

                    negatives.append(
                        {
                            "edited_image_path": neg_image_path,
                            "edited_caption": info["edited_caption"],
                            "category": category,
                            "sub_category": str(len(annotations["bounding_boxes"])),
                            "edit_id": info["edit_id"],
                            "input_phrase": info["input_phrase"],
                            "edited_phrase": info["edited_phrase"],
                        }
                    )

            if not negatives and not region_crop_negatives:
                total_invalid_image_dir += 1
                continue
            curr_neg_data = {
                "image_id": image_id,
                "image_path": image_path,
                "source": "synthetic",
                "caption": annotations["prompt"],
                "negatives": negatives,
            }
            if region_crop_negatives:
                curr_neg_data["region_negatives"] = region_crop_negatives
            training_data.append(curr_neg_data)
            total_negative_samples += len(negatives)
            total_negative_samples += len(region_crop_negatives)

        if enable_sbert:
            embeddings = model.encode([sample["caption"] for sample in non_paired_samples], convert_to_tensor=True)
            for index, sample in tqdm(enumerate(non_paired_samples), desc="Finding Similar Captions"):
                try:
                    query_embedding = model.encode(sample["caption"], convert_to_tensor=True)
                    cos_similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]

                    sorted_indices = cos_similarities.argsort(descending=True).tolist()
                    sorted_indices.remove(index)

                    negatives = []
                    for similar_idx in sorted_indices:
                        # If all are 100% matches (very unlikely but just in case), then skip
                        if cos_similarities[similar_idx].item() >= 0.9:
                            continue

                        negatives.append(
                            {
                                "edited_image_path": non_paired_samples[similar_idx]["image_path"],
                                "edited_caption": non_paired_samples[similar_idx]["caption"],
                                "category": non_paired_samples[similar_idx]["category"],
                            }
                        )
                        if len(negatives) == 1:
                            break

                    total_negative_samples += len(negatives)
                    training_data.append(
                        {
                            "image_path": sample["image_path"],
                            "caption": sample["caption"],
                            "source": "synthetic",
                            "negatives": negatives,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error: {e}")

        logger.info(f"Number of samples => positive: {len(training_data)} negative: {total_negative_samples}")
        logger.info(f"No valid images found directory count => {total_invalid_image_dir}")
        return training_data

    @staticmethod
    def get_image_paths_and_edited_prompt_for_relation(annotations: dict, category: str, image_path: str, threshold=1):
        scores = annotations.get("scores", {})
        if category not in annotations["category"]:
            return None, None

        # check if negative images _negative.png file exists
        edited_image_path = image_path.replace(".png", "_negative.png")
        if not os.path.exists(edited_image_path):
            return None, None

        if (
            scores.get(edited_image_path, {}).get("tifa_score", 0) < threshold
            or scores.get(image_path, {}).get("tifa_score", 0) < threshold
        ):
            return None, None

        if "relation" in category:
            edited_prompt = swap_phrases_in_spatial_prompt(annotations["prompt"], annotations["bounding_boxes"])
        else:
            edited_prompt = annotations.get("edited_prompt")

        if (
            edited_prompt and "\n" in edited_prompt
        ):  # if newline character is present in the prompt, then ignore the sample
            return None, None

        return edited_image_path, edited_prompt

    def process_image_crops_and_create_negative_dict(self, region_crops_info):
        region_crop_negatives = []
        for info in region_crops_info.values():
            bounding_boxes = info.get("bounding_boxes", [])
            if not bounding_boxes:
                continue

            cropped_image = crop_image_region_from_bbox_list(info["image_path"], info["bounding_boxes"])
            crop_id = re.sub(r"\s+|/", "", f"{info['id']}_{info['caption']}")
            crop_fname = f"{crop_id}.png"
            # print(f"crop_fname: {crop_fname}")
            crop_img_path = os.path.join(VIM_DATA_DIR, "region_crop_images", crop_fname)
            if not os.path.exists(crop_img_path):
                cropped_image.save(crop_img_path)

            # add them to the negatives
            region_crop_negatives.append(
                self.format_region_crop_neg_dict(crop_img_path, info["caption"], info["category"], crop_id, True)
            )
        return region_crop_negatives

    def format_region_crop_neg_dict(self, image_path, caption, category, edit_id, region_crop_bool):
        return {
            "edited_image_path": image_path,
            "edited_caption": caption,
            "category": category,
            "edit_id": edit_id,
            "region_crop_bool": region_crop_bool,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_mturk", action="store_true", help="Enable MTurk data preparation")
    parser.add_argument("--enable_region_crop", action="store_true", help="Enable region crop data preparation")
    parser.add_argument("--split", type=str, default="train", help="Split of the data to prepare")
    args = parser.parse_args()

    proc = DataPreparEngine()
    proc.prepare_data(args)

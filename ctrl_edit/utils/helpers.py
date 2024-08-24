import fcntl
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict
from itertools import islice
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.ops import box_convert
from tqdm import tqdm

from commons.constants import (COCO_DATA_DIR, MAIR_LAB_DATA_DIR,
                               SYNTH_DIFFUSE_DATA_DIR, TOTAL_NUM_COCO_CHUNKS)
from commons.logger import Logger

logger = Logger.get_logger(__name__)


def group_outputs_for_batch_repeats(
    all_generated_outputs: list, batch_size: int, repeat_times: int, num_return_sequences: int
):
    grouped_outputs = [[] for _ in range(batch_size)]
    for idx in range(batch_size):
        for repeat_idx in range(repeat_times):
            start_index = idx * num_return_sequences + repeat_idx * (batch_size * num_return_sequences)
            grouped_outputs[idx].extend(all_generated_outputs[start_index : start_index + num_return_sequences])
    return grouped_outputs


def pluralize(noun):
    # Rules for pluralizing nouns in English
    if re.search("[sxz]$", noun) or re.search("[^aeioudgkprt]h$", noun):
        return re.sub("$", "es", noun)
    elif re.search("[^aeiou]y$", noun):
        return re.sub("y$", "ies", noun)
    # Add more rules as needed
    else:
        return noun + "s"


def load_t2icompbench():
    fpath = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", "t2icompbench", "spatial_train.txt")
    logger.info(f"Loading T2I Comparison Benchmark data from {fpath}")
    with open(fpath, "r") as f:
        data = f.read().splitlines()
    return data


def crop_image_region_from_bbox_list(image: Union[str, Image.Image], bboxes: list) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if not bboxes:
        logger.info(f"No bounding boxes found for image. Returning original image.")
        return image

    w, h = image.size
    bboxes = bboxes * np.array([w, h, w, h])

    # increase the bounding box size by 10% to include more context
    bboxes = bboxes * np.array([0.9, 0.9, 1.1, 1.1])

    # Initialize variables for the union region
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), -float("inf"), -float("inf")
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        min_x = min(min_x, x_min)
        min_y = min(min_y, y_min)
        max_x = max(max_x, x_max)
        max_y = max(max_y, y_max)

    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    try:
        # crop the image to the union region
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
    except Exception as e:
        logger.error(f"Error while cropping input image: {e}, returning original image.")
        cropped_image = image

    return cropped_image


# TODO: Need to decide if we want to keep this function or the one below
def annotate_image_with_boxes(image, boxes, phrases):
    """
    Annotate the given image with the provided boxes and associated phrases.

    Parameters:
        image (PIL.Image or np.ndarray): Image to be annotated.
        boxes (list of lists): List of bounding boxes, where each box is represented by four corner points.
        phrases (list of str): List of phrases corresponding to each bounding box.

    Returns:
        PIL.Image: Annotated image.
    """
    if isinstance(image, Image.Image):  # Convert PIL Image to numpy array
        image_np = np.array(image)
    else:
        image_np = image

    h, w, _ = image_np.shape
    boxes_unnorm = boxes * np.array([w, h, w, h])
    # boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    image_with_boxes = image_np.copy()

    for box, phrase in zip(boxes_unnorm, phrases):
        x0, y0, x1, y1 = box
        color = (0, 255, 0)  # Green color in RGB
        thickness = 2  # Line thickness of 2 px
        font_scale = 0.7
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)  # White color in RGB
        bg_color = (0, 0, 0)  # Black color in RGB

        # Draw bounding box
        cv2.rectangle(image_with_boxes, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(phrase, font, font_scale, thickness)

        # Draw background rectangle for the text
        cv2.rectangle(
            image_with_boxes,
            (int(x0), int(y0) - text_height - baseline - 5),
            (int(x0) + text_width, int(y0) - 5),
            bg_color,
            -1,
        )

        # Draw phrase above the bounding box
        cv2.putText(
            image_with_boxes,
            phrase,
            (int(x0), int(y0) - baseline - 5),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return Image.fromarray(image_with_boxes), boxes_unnorm  # Convert back to PIL Image


# def annotate_image_with_boxes(image, boxes, phrases, box_format="xyxy"):
#     """
#     Annotate the given image with the provided boxes and associated phrases.

#     Parameters:
#         image (PIL.Image or np.ndarray): Image to be annotated.
#         boxes (list of lists): List of bounding boxes, where each box is represented by four corner points.
#         phrases (list of str): List of phrases corresponding to each bounding box.

#     Returns:
#         PIL.Image: Annotated image.
#     """
#     if isinstance(image, Image.Image):  # Convert PIL Image to numpy array
#         image_np = np.array(image)
#     else:
#         image_np = image

#     h, w, _ = image_np.shape
#     # first check if the boxes are normalized or not and convert them to unnormalized format
#     if boxes[0][0] < 1:
#         boxes = boxes * np.array([w, h, w, h])
#     if box_format == "xywh":
#         boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#     image_with_boxes = image_np.copy()

#     for box, phrase in zip(boxes, phrases):
#         x0, y0, x1, y1 = box
#         color = (0, 255, 0)  # Green color in RGB
#         thickness = 2  # Line thickness of 2 px
#         font_scale = 0.7
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text_color = (255, 255, 255)  # White color in RGB
#         bg_color = (0, 0, 0)  # Black color in RGB

#         # Draw bounding box
#         cv2.rectangle(image_with_boxes, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)

#         # Get text size
#         (text_width, text_height), baseline = cv2.getTextSize(phrase, font, font_scale, thickness)

#         # Draw background rectangle for the text
#         cv2.rectangle(
#             image_with_boxes,
#             (int(x0), int(y0) - text_height - baseline - 5),
#             (int(x0) + text_width, int(y0) - 5),
#             bg_color,
#             -1,
#         )

#         # Draw phrase above the bounding box
#         cv2.putText(
#             image_with_boxes,
#             phrase,
#             (int(x0), int(y0) - baseline - 5),
#             font,
#             font_scale,
#             text_color,
#             thickness,
#             cv2.LINE_AA,
#         )

#     return Image.fromarray(image_with_boxes), boxes  # Convert back to PIL Image


def load_image_tensor(image: Image.Image) -> torch.Tensor:
    import groundingdino.datasets.transforms as T

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


def get_xyxy_from_xywh(bbox):
    # Assuming bbox format is [x, y, w, h]
    x0, y0, w, h = bbox
    return x0, y0, x0 + w, y0 + h


def remove_duplicate_dict_entries_by_key(data_list, key):
    seen_values = set()
    unique_data = []
    for data in data_list:
        if data[key] not in seen_values:
            unique_data.append(data)
            seen_values.add(data[key])
    return unique_data


def copy_current_cache_file_as_backup_json(cache_file_path):
    backup_file_path = cache_file_path.replace(".json", f"_backup.json")

    if os.path.exists(cache_file_path):
        shutil.copy2(cache_file_path, backup_file_path)
        logger.info(f"Cache file backed up at {backup_file_path}")
    else:
        logger.warning(f"Cache file {cache_file_path} does not exist, no backup created")


def remove_current_cache_backup_file(cache_file_path):
    backup_file_path = cache_file_path.replace(".json", f"_backup.json")
    if os.path.exists(backup_file_path):
        os.remove(backup_file_path)
        logger.info(f"Cache file removed at {backup_file_path}")
    else:
        logger.warning(f"Cache file {backup_file_path} does not exist, no removal needed")


def get_coco_path_by_image_id(split: str, image_id: Union[str, int]):
    image_id_str = f"{int(image_id):012d}"
    return os.path.join(MAIR_LAB_DATA_DIR, "coco", "images", f"{split}2017", f"{image_id_str}.jpg")


def get_unique_coco_annotations(split: str):
    fpath = os.path.join(MAIR_LAB_DATA_DIR, "coco", "annotations", f"captions_{split}2017.json")
    logger.info(f"Loading COCO captions from {fpath}")
    with open(fpath, "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())

    annotations = json_data["annotations"]
    random.shuffle(annotations)

    # keep only one caption per image
    unique_image_ids = []
    new_annotations = []
    for entry in tqdm(annotations, desc="Removing duplicate captions"):
        if entry["image_id"] not in unique_image_ids:
            unique_image_ids.append(entry["image_id"])
            new_annotations.append({"image_id": entry["image_id"], "caption": entry["caption"]})

    return new_annotations


def load_coco_captions(split: str, chunk_index: int = None, force_chunking=False):
    if split == "validation":
        split = "val"
    unique_annotaion_fpath = os.path.join(
        SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"unique_coco_annotations_{split}2017.json"
    )

    # If the unique annotations file does not exist, create it and save it
    if not os.path.exists(unique_annotaion_fpath):
        logger.info(f"Unique COCO captions file not found. Creating {unique_annotaion_fpath}")
        annotations = get_unique_coco_annotations(split)
        with open(unique_annotaion_fpath, "w") as f:
            json.dump(annotations, f, indent=4)
    else:
        logger.info(
            f"Loading unique COCO captions from {unique_annotaion_fpath}. "
            "Please note that if a chunk_index is provided, this data may be replaced "
            "by the corresponding chunked data."
        )
        annotations = load_json_data(unique_annotaion_fpath)

    # Now, let's see if the process is requesting a chunk of the data
    if chunk_index is not None and isinstance(chunk_index, int):
        # The original filename becomes a directory with "_chunked" suffix. Chunks are saved there by index.
        unique_annotaion_chunk_fpath = unique_annotaion_fpath.replace(".json", f"_chunked/chunk_{chunk_index}.json")
        if not os.path.exists(unique_annotaion_chunk_fpath):
            if force_chunking:
                logger.warning(
                    "You are creating new chunks. This action may have consequences. "
                    "Subsequent scripts might have already processed the existing chunks. "
                    "Creating new chunks could lead to inconsistencies in the data processing pipeline."
                )
                os.makedirs(os.path.dirname(unique_annotaion_chunk_fpath), exist_ok=True)
                # This means we have to chunk the whole file and save all the chunks
                logger.info(f"Chunking the unique COCO captions file into {TOTAL_NUM_COCO_CHUNKS} chunks.")
                chunk_size = len(annotations) // TOTAL_NUM_COCO_CHUNKS
                all_chunks = return_all_chunks_of_data(annotations, chunk_size)
                for i, chunk in enumerate(all_chunks):
                    curr_chunk_fpath = unique_annotaion_fpath.replace(".json", f"_chunked/chunk_{i}.json")
                    save_to_json(chunk, curr_chunk_fpath)
            else:
                raise ValueError(
                    f"You are attempting to access a chunk of data but the requested file {unique_annotaion_chunk_fpath} does not exist. "
                    "Chunking is a sensitive operation that should typically be performed only once "
                    "to prevent inconsistencies in the data processing pipeline. "
                    "If you understand the implications and wish to proceed, "
                    "set the `force_chunking` parameter to `True`."
                )
        logger.info(f"Loading chunk {chunk_index} of unique COCO captions from {unique_annotaion_chunk_fpath}")
        annotations = load_json_data(unique_annotaion_chunk_fpath)

    # if split == "train":
    #     from .skill_terms import SkillAnalyzer

    #     skill_ann = SkillAnalyzer()
    #     annotations = skill_ann.get_skill_only_data(annotations)

    if split == "val":
        logger.info("Loading only attribute validation data.")
        attribute_only_fpath = "/network/projects/aishwarya_lab/members/rabiul/prompt_resources/llm_edits_coco/openai/is_attribute_validation.json"
        with open(attribute_only_fpath, "r") as f:
            attribute_ann = json.load(f)
            """
            "13348": {
                "A large plane is parked at the airport.": {
                    "raw_response": "{\"id\": \"chatcmpl-9NZrGJRtGAd4J00d2lrFvMGQhs66Y\", \"choices\": [{\"finish_reason\": \"stop\", \"index\": 0, \"logprobs\": null, \"message\": {\"content\": \"Input: A large plane is parked at the airport.\\nAttributeExist: yes\", \"role\": \"assistant\", \"function_call\": null, \"tool_calls\": null}}], \"created\": 1715406410, \"model\": \"gpt-3.5-turbo-1106\", \"object\": \"chat.completion\", \"system_fingerprint\": null, \"usage\": {\"completion_tokens\": 15, \"prompt_tokens\": 358, \"total_tokens\": 373}}",
                    "output": "yes"
                }
            },
            """
            new_annotations = []
            for image_id, ann in attribute_ann.items():
                for caption, data in ann.items():
                    if data["output"] == "yes":
                        new_annotations.append({"image_id": int(image_id), "caption": caption})
            annotations = new_annotations

    logger.info(f"Successfully loaded {len(annotations)} unique COCO captions from {unique_annotaion_fpath}")

    return annotations


def load_vsr_captions(split="validation"):
    from datasets import load_dataset

    dataset = load_dataset("cambridgeltl/vsr_random")[split]

    new_dataset = []
    for entry in dataset:
        if not entry["label"]:
            continue
        image_id = entry["image"].replace(".jpg", "")
        coco2017train_dir = "/network/projects/aishwarya_lab/datasets/coco/images/train2017/"
        img_path = os.path.join(coco2017train_dir, entry["image"])
        if "val" in entry["image_link"]:
            img_path = img_path.replace("train", "val")
        new_dataset.append({"image_id": image_id, "caption": entry["caption"], "image_path": img_path})

    return new_dataset


def load_data_from_cc12m():
    dataset_directory = os.path.join(MAIR_LAB_DATA_DIR, "cc12m")
    file_path = os.path.join(dataset_directory, "cc12m.tsv")
    limited_dataframe = pd.read_csv(file_path, sep="\t", nrows=2000000, names=["url", "caption"])
    return limited_dataframe


def get_less_frquent_relation_types_from_vsr_data():
    from datasets import load_dataset

    def count_relations_in_captions(unique_relations, annotations):
        """Count occurrences of each relation in the captions."""
        relation_counts = defaultdict(int)
        for entry in tqdm(annotations, desc="Counting relations in captions"):
            caption_text = entry["caption"]
            for relation in unique_relations:
                if relation in caption_text:
                    relation_counts[relation] += 1
                    break
        return relation_counts

    # Load VSR dataset and COCO captions
    vsr_dataset = load_dataset("cambridgeltl/vsr_random")["validation"]
    coco_annotations = load_coco_captions(split="train")

    # Get unique relations and count their occurrences in COCO captions
    unique_relations = set(entry["relation"] for entry in vsr_dataset)
    relation_counts = count_relations_in_captions(unique_relations, coco_annotations)

    # Sort and filter the relation counts
    sorted_relation_counts = dict(sorted(relation_counts.items(), key=lambda item: item[1], reverse=True))
    filtered_relation_counts = {relation: count for relation, count in sorted_relation_counts.items() if count < 1000}

    filtered_relation = list(filtered_relation_counts.keys())

    fname = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", "filtered_vsr_relation_types.txt")
    with open(fname, "w") as f:
        for relation in filtered_relation:
            f.write(relation + "\n")
    print(f"Filtered vsr relations saved to {fname}")


def attatch_caption_to_image(image, title_text: str, image_id=None):
    """
    image: the generated image
    title_text: given prompt to be inserted on top of the image
    image_id: an unique identifier to be used for file saving
    """
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 16)
    image_editable = ImageDraw.Draw(image)
    image_editable.text((15, 15), title_text, (237, 230, 211), font=font)
    return image


def resize_maintaining_aspect_ratio(image: Union[Image.Image, np.ndarray], target_size=256):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size
    aspect_ratio = width / height

    new_width = target_size if aspect_ratio > 1 else int(target_size * aspect_ratio)
    new_height = int(target_size / aspect_ratio) if aspect_ratio > 1 else target_size

    return image.resize((new_width, new_height)), width, height


def resize_image(image: Union[Image.Image, np.ndarray], width, height):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError("Provided input is neither a valid PIL Image object nor a numpy ndarray.")

    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    return image


def transform_image(image_source: Image.Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            # T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.Resize((512, 512)),  # Add this line to resize the image
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def load_subset_of_data_by_chunk(data, chunk_size, chunk_index):
    if isinstance(data, list):
        # Handle list: Use slicing to return a chunk of the list.
        start_index = chunk_size * chunk_index
        end_index = chunk_size * (chunk_index + 1)
        return data[start_index:end_index]
    elif isinstance(data, dict):
        # Handle dictionary: Construct a new dictionary with a subset of key-value pairs.
        keys = list(data.keys())
        start_index = chunk_size * chunk_index
        end_index = chunk_size * (chunk_index + 1)
        subset_keys = keys[start_index:end_index]
        return {key: data[key] for key in subset_keys}
    else:
        # Handle unsupported data types.
        raise ValueError("Unsupported data type. Expected list or dictionary.")


def return_all_chunks_of_data(data, chunk_size):
    if isinstance(data, list):
        # Handle list: Use slicing to return a chunk of the list.
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    elif isinstance(data, dict):
        # Handle dictionary: Use slicing to return a chunk of the dictionary.
        keys = list(data.keys())
        return [{key: data[key] for key in islice(keys, i, i + chunk_size)} for i in range(0, len(data), chunk_size)]
    else:
        # Handle unsupported data types.
        raise ValueError("Unsupported data type. Expected list or dictionary.")


def get_coco_tags(split: str):
    # Downloaded from https://cs.stanford.edu/people/karpathy/deepimagesent
    with open(os.path.join(COCO_DATA_DIR, "annotations", f"instances_{split}2017.json")) as f:
        instances = json.load(f)
        annotations = instances["annotations"]
        images = instances["images"]
        tag_categories = instances["categories"]
    # id is segmentation_id
    # {"segmentation": [[239.97,260.24,222.04,270.49,199.84,253.41,213.5,227.79,259.62,200.46,274.13,202.17,277.55,210.71,249.37,253.41,237.41,264.51,242.54,261.95,228.87,271.34]],"area": 2765.1486500000005,"iscrowd": 0,"image_id": 558840,"bbox": [199.84,200.46,77.71,70.88],"category_id": 58,"id": 156}
    imageid_dimensions = defaultdict()
    imageid_tags = defaultdict(list)
    for image in images:
        image_id = image["id"]
        width = image["width"]
        height = image["height"]
        imageid_dimensions[image_id] = (width, height)

    for ann in annotations:
        image_id = ann["image_id"]
        area = ann["area"]
        category_id = ann["category_id"]
        id = ann["id"]
        bbox = ann["bbox"]
        segmentation = ann["segmentation"]
        imageid_tags[image_id].append(
            {"id": id, "category_id": category_id, "area": area, "bbox": bbox, "segmentation": segmentation}
        )

    num_tags = []
    for id in imageid_tags.keys():
        num_tags.append(len(imageid_tags[id]))

    categories_id_name = defaultdict()
    categories_name_id = defaultdict()
    categories_id_to_supercategory = defaultdict()
    for category in tag_categories:
        id = category["id"]
        name = category["name"]
        supercategory = category["supercategory"]
        categories_id_to_supercategory[id] = supercategory
        categories_id_name[id] = name
        categories_name_id[name] = id

    return imageid_tags, imageid_dimensions, categories_name_id, categories_id_to_supercategory

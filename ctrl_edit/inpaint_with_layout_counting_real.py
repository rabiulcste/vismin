import argparse
import json
import os
import random
import re
import time
import uuid
from collections import Counter, defaultdict
from typing import List

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm

from commons.constants import (COCO_DATA_DIR, LANGUAGE_MODEL_NAMES,
                               MISTRALAI_LANGUAGE_MODEL_NAMES,
                               SYNTH_DIFFUSE_DATA_DIR, VIM_DATA_DIR)
from commons.logger import Logger

from .filters.object_detection import ObjectGroundingDetector
from .filters.vqa_models import VQAModelForTifa
from .utils.helpers import (copy_current_cache_file_as_backup_json,
                            get_coco_path_by_image_id, get_coco_tags,
                            load_coco_captions, load_json_data,
                            save_array_to_img, save_to_json, set_random_seed)
from .utils.llm_utils import BaseLLM
from .utils.rmvany_utils import (ObjectCountParser,
                                 get_new_caption_for_updated_count,
                                 get_obj_removal_verify_qa_pairs_by_name,
                                 get_updated_count_phrase)
from .utils.spatial_relation_utils import (
    load_llama_model_for_inpainting,
    remove_object_from_image_using_llama_model)

logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed()


class GenerateEditInstructionCounting(BaseLLM):
    EDIT_PATTERN = "regex pattern to match the edit instruction"  # : add regex pattern to match the edit instruction

    def set_cache_file_path(self, cache_file_path):
        self.cache_file_path = cache_file_path

    def get_prompt_data_fpath(self, prompt_type):
        return os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"{prompt_type}.json")

    @staticmethod
    def _load_cached_data(json_file):
        if os.path.exists(json_file):
            logger.info(f"Loading existing cache from {json_file}")
            with open(json_file, "r") as f:
                return json.load(f)

        return defaultdict(dict)

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int):
        examples = self.prompt_data["samples"]

        random.shuffle(examples)
        selected_examples = random.sample(examples, num_examples_in_task_prompt)

        # capitalize the first letter of the caption and add a period at the end if it doesn't exist
        if image_caption:
            image_caption = image_caption[0].upper() + image_caption[1:]
        if not image_caption.endswith("."):
            image_caption += "."

        # Create the final prompt instruction with correct numbering
        prompt_instruction = self.prompt_data["task_instruction"]
        for i, example in enumerate(selected_examples, start=1):
            if isinstance(example, dict):
                example = " ".join([f"{k}: {v}" for k, v in example.items()])
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{image_caption}} {{output_key}}:"

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            image_caption=image_caption,
            output_key=self.prompt_data["io_structure"]["output_keys"],
        )
        if self.language_model_name == "mistralai/Mistral-7B-Instruct-v0.2":
            prompt = "[INST]" + prompt + "[/INST]"
        return prompt

    def generate_counting_edits_from_caption(
        self,
        input_texts: List[str],
        max_length: int = 256,
        num_examples_in_task_prompt: int = 8,
    ) -> str:
        all_prompts = [self._prepare_prompt(text, num_examples_in_task_prompt) for text in input_texts]
        logger.debug(f"all prompts: {json.dumps(all_prompts, indent=2)}")

        generated_outputs = self.generate_output_batch(
            all_prompts,
            max_length=max_length,
            num_examples_in_task_prompt=num_examples_in_task_prompt,
        )
        logger.debug(f"Generated outputs: {json.dumps(generated_outputs, indent=2)}")

        for input_text, output_text in zip(input_texts, generated_outputs):
            logger.info(f"{input_text} => {output_text}")

        return generated_outputs


class CountingEditsPreprocessor:
    def __init__(
        self,
        split: str,
        output_data_dir: str,
    ):
        self.split = split
        self.output_data_dir = output_data_dir
        self.imageid_tags, self.imageid_dimensions, self.categories_name_id = get_coco_tags(split)
        self.load_models()

    def load_models(self):
        logger.info("Starting the process of loading models.")

        logger.info("Initializing OwlV2Detector.")
        objdet_engine = ObjectGroundingDetector()
        logger.info("OwlV2Detector initialized successfully.")

        logger.info("Initializing ObjectCountParser.")
        obj_cnt_parser = ObjectCountParser()
        logger.info("ObjectCountParser initialized successfully.")

        self.objdet_engine = objdet_engine
        self.obj_cnt_parser = obj_cnt_parser

        logger.info("Models loaded successfully.")

    @staticmethod
    def create_mask_image_from_bbox(bbox, image_height, image_width):
        bbox_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        x_min, y_min, width, height = bbox
        x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
        bbox_mask[y_min : y_min + height, x_min : x_min + width] = 255

        return bbox_mask

    def create_remove_coco_instances_suggestions(self, image_id, caption_text: str, object_info: dict):
        image_width, image_height = self.imageid_dimensions[int(image_id)]
        image_tags = self.imageid_tags[int(image_id)]
        obj_name, obj_cnt_n, obj_cnt_t, obj_id = object_info
        new_count_n, new_count_t, original_count_phrase, updated_count_phrase = get_updated_count_phrase(
            obj_name, obj_cnt_n, obj_cnt_t
        )
        new_caption_text = get_new_caption_for_updated_count(caption_text, original_count_phrase, updated_count_phrase)

        curr_edit_data = []
        # trying to find the tag for the object in COCO annotations
        for tag in image_tags:
            if tag["category_id"] != obj_id:
                continue

            bbox = tag["bbox"]
            segmentation = tag["segmentation"]

            logger.debug(
                f"id: {tag['id']} # object name: {obj_name}, object count: {obj_cnt_n}, reduced count: {new_count_n}"
            )
            logger.info(f"(coco instances) old caption: {caption_text} => new caption: {new_caption_text}")

            try:
                # Create a blank mask
                mask = np.zeros((image_height, image_width), dtype=np.uint8)

                # If segmentation is a list of polygons
                if isinstance(segmentation[0], list):
                    for polygon in segmentation:
                        # Reshape the flat list into a list of (x, y) points
                        polygon_points = np.array(polygon, dtype=np.int32).reshape(-1, 2)

                        # Draw the polygon on the image
                        mask = cv2.fillPoly(mask, [polygon_points], color=(255, 255, 255))  # White color
                else:
                    # If segmentation is a flat list of coordinates
                    polygon_points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)

                    # Draw the polygon on the image
                    mask = cv2.fillPoly(mask, [polygon_points], color=(255, 255, 255))  # White color
            except Exception as e:
                logger.info(f"Error: {e}")
                continue

            # create a mask image based on bounding box
            bbox_mask = self.create_mask_image_from_bbox(bbox, image_height, image_width)

            # Save or display the modified image
            # mask_image_path = os.path.join(self.output_data_dir, str(image_id), f"{tag['id']}_mask.png")
            bbox_mask_image_path = os.path.join(self.output_data_dir, str(image_id), f"{tag['id']}_bbox_mask.png")
            os.makedirs(os.path.dirname(bbox_mask_image_path), exist_ok=True)
            # cv2.imwrite(mask_image_path, mask)  # Saves the modified image
            cv2.imwrite(bbox_mask_image_path, bbox_mask)  # Saves the modified image
            logger.info(f"saved images at: {bbox_mask_image_path}")
            curr_edit_data.append(
                {
                    "id": tag["id"],
                    "bbox": tag["bbox"],
                    "segmentation": segmentation,
                    "object_name": obj_name,
                    "new_caption": new_caption_text,
                    "original_object_count": obj_cnt_n,
                    "reduced_object_count": new_count_n,
                }
            )

            if len(curr_edit_data) > 2:  # don't process more than 2 objects
                break

        return curr_edit_data

    def create_remove_open_ended_suggestions(
        self, image_id: str, image_path: str, caption_text: str, object_info: dict
    ):
        image_width, image_height = self.imageid_dimensions[int(image_id)]
        obj_name, obj_cnt_n, obj_cnt_t = object_info
        new_count_n, new_count_t, original_count_phrase, updated_count_phrase = self.get_updated_count_phrase(
            obj_name, obj_cnt_n, obj_cnt_t
        )
        new_caption_text = self.get_new_caption_for_updated_count(
            caption_text, original_count_phrase, updated_count_phrase
        )
        detection_failed = False

        # Initial detection and counting
        object_counts, detections = self.objdet_engine.generate_grounding(
            image_path, obj_name, box_threshold=0.4, text_threshold=0.25
        )
        logger.debug(f"(G-DINO) {object_counts} => {detections}")

        # Verify if the object is detected and the count is correct in the first detection
        if obj_name not in object_counts or object_counts[obj_name] != obj_cnt_n:
            logger.debug(
                f"object: {obj_name} not detected or count mismatched in first detection, image: {object_counts.get(obj_name, 'None')}, text: {obj_cnt_n}"
            )
            detection_failed = True

        # Additional detection and counting with specific thresholds
        if detection_failed:
            object_counts, detections = self.objdet_engine._detect_and_count_objects_in_image(
                image_path, [obj_name], threshold=0.1, nms_threshold=0.6
            )
            logger.debug(f"(Owlv2) {object_counts} => {detections}")

            # Verify if the object is detected and the count is correct in the second detection
            if obj_name not in object_counts or object_counts[obj_name] != obj_cnt_n:
                logger.debug(
                    f"object: {obj_name} not detected or count mismatched in second detection, image: {object_counts.get(obj_name, 'None')}, text: {obj_cnt_n}"
                )
                detection_failed = True

        if detection_failed:
            return []

        logger.info(f"(open ended) old caption: {caption_text} => new caption: {new_caption_text}")

        curr_edit_data_list = []
        for detection in detections[obj_name]:
            tag_id = str(uuid.uuid4())[:8]
            bbox = detection["location"]
            # crop the detected object from the image
            cropped_object = Image.open(image_path).crop(bbox)
            cropped_object = cropped_object.resize((512, 512)).convert("RGB")
            # let's detect the object on this crop, to verify if it just one object
            object_counts, _ = self.owlv2._detect_and_count_objects_in_image(
                cropped_object, [obj_name], threshold=0.05, nms_threshold=0.6
            )
            if obj_name not in object_counts or object_counts[obj_name] != 1:
                continue

            # create a mask image based on bounding box
            bbox_mask = self.create_mask_image_from_bbox(bbox, image_height, image_width)

            # blend the mask with the original image
            og_mask_blend_image = cv2.addWeighted(
                cv2.imread(image_path), 0.7, cv2.cvtColor(bbox_mask, cv2.COLOR_GRAY2BGR), 0.3, 0
            )

            og_mask_blend_image_path = os.path.join(self.output_data_dir, str(image_id), f"{tag_id}_mask_blend.png")
            bbox_mask_image_path = os.path.join(self.output_data_dir, str(image_id), f"{tag_id}_bbox_mask.png")
            os.makedirs(os.path.dirname(og_mask_blend_image_path), exist_ok=True)
            cv2.imwrite(og_mask_blend_image_path, og_mask_blend_image)  # Saves the modified image
            cv2.imwrite(bbox_mask_image_path, bbox_mask)  # Saves the modified image
            logger.info(f"saved images (object-detect) at: {bbox_mask_image_path}\n {og_mask_blend_image_path}")

            curr_edit_data = {
                "id": tag_id,  # unique id for the edit
                "bbox": bbox,
                "object_name": obj_name,
                "new_caption": new_caption_text,
                "original_object_count": obj_cnt_n,
                "reduced_object_count": new_count_n,
            }
            curr_edit_data_list.append(curr_edit_data)
        return curr_edit_data_list

    @staticmethod
    def word_frequency(text):
        # Split the text into words using split() method
        words = text.split()
        # Use Counter to count the frequency of each word in the list
        word_count = Counter(words)
        return word_count

    def create_masked_annotation_for_coco_data(self, split: str):
        annotations = load_coco_captions(split)
        random.shuffle(annotations)
        cntr, total_coco_instances, total_open_ended = 0, 0, 0

        already_processed_images = os.listdir(self.output_data_dir)
        already_processed_images = [
            str(image_id)
            for image_id in already_processed_images
            if os.path.exists(os.path.join(self.output_data_dir, str(image_id), "annotations.json"))
        ]
        already_processed_images = set(already_processed_images)
        logger.info(f"Annotation found in cache: {len(already_processed_images)} images")

        existence_object_edit_count = 0
        for entry in tqdm(annotations, desc="processing coco dataset"):
            caption_text = entry["caption"]
            image_id = entry["image_id"]
            if str(image_id) in already_processed_images:
                continue

            annotations_fpath = os.path.join(self.output_data_dir, str(image_id), "annotations.json")
            image_path = get_coco_path_by_image_id(split, image_id)

            apply_a_to_one = False
            if existence_object_edit_count < 10 and random.random() > 0.9999:
                apply_a_to_one = True
                existence_object_edit_count += 1

            noun_chunks, caption_text_updated = self.obj_cnt_parser._parse_objects_from_caption(
                caption_text, apply_a_to_one
            )
            objects_to_detect = [self.obj_cnt_parser._extract_count_from_chunk(chunk) for chunk in noun_chunks]
            object_freq_in_sentence = self.word_frequency(caption_text)
            objects_to_detect = [
                (object_name, count_n, count_t)
                for object_name, count_n, count_t in objects_to_detect
                if count_n is not None and object_name and object_freq_in_sentence.get(object_name) == 1
            ]
            object_to_detect_tagged = [
                (obj_name, count_n, count_t, self.categories_name_id.get(obj_name))
                for obj_name, count_n, count_t in objects_to_detect
                if self.categories_name_id.get(obj_name) is not None
            ]

            if object_to_detect_tagged:
                # If the object count is 1 and the object name is preceded by "a" or "an" in the caption,
                # we'll replace "a" or "an" with "one" for clarity.
                for object_info in object_to_detect_tagged:
                    obj_name, obj_cnt_n, obj_cnt_t, obj_id = object_info
                    if obj_cnt_n == 1:
                        # replace "a" or "an" with "one" only when it precedes the object name
                        caption_text = re.sub(
                            r"\b(a|an)\s+" + re.escape(obj_name) + r"\b",
                            "one " + obj_name,
                            caption_text,
                            flags=re.IGNORECASE,
                        )
                logger.info(f"prompt: {caption_text}")
                logger.info(f"objects to detect tagged: {object_to_detect_tagged}")

            image_edit_source = None
            coco_instances_removal_data = []
            for object_info in object_to_detect_tagged:
                logger.debug(f"processing: {entry}")
                logger.debug(f"objects detected: {object_info}")
                curr_edit_instructions = self.create_remove_coco_instances_suggestions(
                    image_id, caption_text, object_info
                )
                coco_instances_removal_data.extend(curr_edit_instructions)

            if coco_instances_removal_data:
                image_edit_source = "coco_annotations"
                total_coco_instances += len(coco_instances_removal_data)

            open_ended_removal_data = []
            if image_edit_source is None:
                # since remove or replace is not possible from COCO ground truth annotations, we will approach it using object detection
                for object_info in objects_to_detect:
                    curr_edit_instruction = self.create_remove_open_ended_suggestions(
                        image_id, image_path, caption_text, object_info
                    )
                    open_ended_removal_data.extend(curr_edit_instruction)

            if open_ended_removal_data:
                image_edit_source = "open_ended"
                total_open_ended += len(open_ended_removal_data)

            if image_edit_source is None:
                continue

            os.makedirs(os.path.dirname(annotations_fpath), exist_ok=True)
            with open(annotations_fpath, "w") as f:
                json.dump(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "input_caption": caption_text,
                        "edit_instructions": coco_instances_removal_data or open_ended_removal_data,
                        "image_edit_source": image_edit_source,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved annotations at: {annotations_fpath}")
            cntr += 1

            if (cntr + 1) % 50 == 0:
                logger.warning(f"********** Processed {cntr + 1} images **********")
                logger.warning(f"********** Total COCO instances: {total_coco_instances} **********")
                logger.warning(f"********** Total open ended: {total_open_ended} **********")

    def free_models_from_gpu_memory(self):
        del self.objdet_engine
        del self.obj_cnt_parser

    def filter_edit_instructions(self):
        already_processed_images = os.listdir(self.output_data_dir)
        already_processed_images = [
            str(image_id)
            for image_id in already_processed_images
            if os.path.exists(os.path.join(self.output_data_dir, str(image_id), "annotations.json"))
        ]
        already_processed_images = list(set(already_processed_images))
        random.shuffle(already_processed_images)
        logger.info(f"Annotation found in cache: {len(already_processed_images)} images")

        model_name_or_path = "liuhaotian/llava-v1.6-34b"  # or liuhaotian/llava-v1.6-34b or liuhaotian/llava-v1.5-13b
        vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=8)

        object_count_dict = defaultdict(int)
        for image_id in tqdm(already_processed_images, desc="filtering edit instructions"):
            annotations_fpath = os.path.join(self.output_data_dir, str(image_id), "annotations.json")
            with open(annotations_fpath, "r") as f:
                annotations = json.load(f)

            for info in annotations["edit_instructions"]:
                object_count_dict[info["original_object_count"]] += 1

            if "background_description" in annotations:
                logger.debug(f"skipping image: {annotations_fpath} as it's already processed")
                continue
            image_path = annotations["image_path"]
            logger.info(f"processing: {annotations_fpath}")
            image = Image.open(image_path).convert("RGB")
            prompt = "Describe only the background of this image. Do not describe any objects in the image."
            background_description = vqa_scorer.get_output(prompt, image)
            # only keep the first sentence and remove the phrase "The background of the image features"
            background_description = background_description.split(".")[0]
            background_description = re.sub(r"The background of the image features", "", background_description).strip()

            annotations["background_description"] = background_description
            logger.info(f"background description: {background_description}")

            image_edit_source = annotations["image_edit_source"]
            edit_instructions = annotations["edit_instructions"]

            was_removed = False
            if image_edit_source == "open_ended":
                for edit_info in annotations["edit_instructions"][:]:
                    obj_name = edit_info["object_name"]
                    bbox = edit_info["bbox"]
                    cropped_object = image.crop(bbox)
                    # resize the cropped object to 224x224
                    cropped_object = cropped_object.resize((512, 512))
                    question_answer_pairs = [
                        {
                            "question": f"How many {obj_name} are in the image?",
                            "choices": ["one", "two", "three"],
                            "answer": "one",
                        },
                        {"question": f"How many {obj_name} are there?", "choices": ["1", "2", "3"], "answer": "1"},
                        # {"question": f"Is there only one {obj_name}?", "choices": ["yes", "no"], "answer": "yes"},
                        # {"question": f"Is there more than one {obj_name}?", "choices": ["yes", "no"], "answer": "no"},
                        {"question": f"Is the {obj_name} fully visible?", "choices": ["yes", "no"], "answer": "yes"},
                    ]
                    tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, cropped_object)
                    logger.info(f"tifa results: {tifa_result_dict['tifa_score']}")
                    edit_info["tifa_score"] = tifa_result_dict["tifa_score"]
                    # if score is not 1 then remove the edit instruction
                    if tifa_result_dict["tifa_score"] < 0.9:
                        edit_instructions.remove(edit_info)
                        was_removed = True
                        # remove the mask and blend images
                        bbox_mask_image_path = os.path.join(
                            self.output_data_dir, str(image_id), f"{edit_info['id']}_bbox_mask.png"
                        )
                        og_mask_blend_image_path = os.path.join(
                            self.output_data_dir, str(image_id), f"{edit_info['id']}_mask_blend.png"
                        )
                        if os.path.exists(bbox_mask_image_path):
                            os.remove(bbox_mask_image_path)
                        if os.path.exists(og_mask_blend_image_path):
                            os.remove(og_mask_blend_image_path)

            if was_removed:
                annotations["edit_instructions"] = edit_instructions

            with open(annotations_fpath, "w") as f:
                json.dump(annotations, f, indent=2)
            logger.info(f"Saved updated annotations at: {annotations_fpath}")

        logger.info("********** START OBJECT COUNT REPORT **********")
        logger.info(f"Total object count: {dict(object_count_dict)}")
        logger.info("*********** END OBJECT COUNT REPORT ***********")

    # TODO: add support for countbench dataset
    def create_masked_annotation_for_countbench_data(self, split):
        pass


def generate_edit_instructions_for_counting(split: str, output_data_dir: str):
    proc = CountingEditsPreprocessor(split, output_data_dir)
    proc_func = getattr(proc, f"create_masked_annotation_for_{dataset}_data")
    proc_func(split)
    proc.free_models_from_gpu_memory()
    proc.filter_edit_instructions()


def get_resized_bbox(bbox: list, scale_x: float, scale_y: float):
    resized_bbox = [
        int(bbox[0] * scale_x),
        int(bbox[1] * scale_y),
        int(bbox[2] * scale_x),
        int(bbox[3] * scale_y),
    ]

    return resized_bbox


def get_segmented_image(image: Image.Image, segmentation):
    if segmentation is None:
        return image

    image_width, image_height = image.size
    # convert to cv2 format
    image = np.array(image)
    try:
        # Ensure the input image is in the correct format (H, W, C)
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create a mask of the same size as the image, filled with zeros (black)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # If segmentation is a list of polygons
        if isinstance(segmentation[0], list):
            for polygon in segmentation:
                # Reshape the flat list into a list of (x, y) points
                polygon_points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                # Draw the polygon on the mask with white color
                cv2.fillPoly(mask, [polygon_points], color=1)
        else:
            # If segmentation is a flat list of coordinates
            polygon_points = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
            # Draw the polygon on the mask with white color
            cv2.fillPoly(mask, [polygon_points], color=1)

        # Convert mask to boolean
        mask = mask.astype(bool)

        # Create an output image filled with white pixels
        segmented_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        # Copy pixels from the original image where the mask is True
        segmented_image[mask] = image[mask]

    except Exception as e:
        print(f"Error: {e}")
        return image

    return Image.fromarray(segmented_image)


def verify_already_generated_but_unfiltered_images(vqa_scorer: VQAModelForTifa, image_path: str, annotation: dict):
    logger.info(f"Verifying already generated but unfiltered image: {image_path}")
    bbox = annotation["bbox"]
    object_name = annotation["object_name"]

    inpainted_image = load_image(image_path)
    segmentation = annotation.get("segmentation", None)
    # get the segmentated image using the segmentation mask and treat it as the inpainted image
    segmented_image = get_segmented_image(inpainted_image, segmentation)
    cropped_object = inpainted_image.crop(
        (
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3],
        )
    )
    # cropped_object.save(f"diffused_generator/output/counting/{annotation['id']}_cropped.png")
    # inpainted_image.save(f"diffused_generator/output/counting/{annotation['id']}_inpainted.png")
    question_answer_pairs = get_obj_removal_verify_qa_pairs_by_name(object_name)
    tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, cropped_object)
    logger.info(f"tifa results: {tifa_result_dict['tifa_score']}")
    annotation["tifa_score"] = tifa_result_dict["tifa_score"]


def run_llama_removal_for_counting(split: str, output_data_dir: str, debug: bool = False):
    llama_model, inpaint_img_with_builded_lama = load_llama_model_for_inpainting()

    tot_proc_images = 0
    image_directories = os.listdir(output_data_dir)
    for image_id in tqdm(image_directories, desc="inpainting images"):
        inp_image_path = get_coco_path_by_image_id(split, image_id)
        json_path = os.path.join(output_data_dir, image_id, "annotations.json")

        data = load_json_data(json_path)
        if data is None:
            continue

        input_image = Image.open(inp_image_path).convert("RGB")
        for edit_info in data["edit_instructions"]:
            bbox = edit_info["bbox"]
            object_name = edit_info["object_name"]
            object_roi_image = remove_object_from_image_using_llama_model(
                llama_model, input_image, [bbox], inpaint_img_with_builded_lama, dilate_kernel_size=5
            )
            tot_proc_images += 1

        if tot_proc_images > 10:
            break


def run_diffusion_inpainting_for_counting(split: str, output_data_dir: str, debug: bool = False):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    # pipe.to(device)

    model_name_or_path = "liuhaotian/llava-v1.6-34b"  # or liuhaotian/llava-v1.6-34b or liuhaotian/llava-v1.5-13b
    vqa_scorer = VQAModelForTifa(model_name_or_path)

    image_directories = os.listdir(output_data_dir)
    random.shuffle(image_directories)
    logger.info(f"Total images to process: {len(image_directories)}")

    object_count_dict = defaultdict(int)
    EXISTENCE_OBJECT_EDIT_COUNT_LIMIT = 10
    for image_id in tqdm(image_directories, desc="inpainting images"):
        inp_image_path = get_coco_path_by_image_id(split, image_id)
        inp_image = load_image(inp_image_path)
        # resize the input image and mask to 1024x1024
        inp_image_rsz = inp_image.resize((1024, 1024))

        # load prompt from the new caption
        json_path = os.path.join(output_data_dir, image_id, "annotations.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)

        background_description = data.get("background_description")
        if background_description is None:
            logger.info(f"skipping image: {image_id} as background description is not found")
            continue

        was_changed = False
        for edit_info in data["edit_instructions"]:
            img_inpainted_path = os.path.join(output_data_dir, image_id, f"{edit_info['id']}_diffusion_inpainted.png")
            if os.path.exists(img_inpainted_path):
                if "tifa_score" not in edit_info:
                    if edit_info["original_object_count"] == 1:
                        # Special case: reducing original count results in object existence (zero count) image-text pair.
                        # To prevent over-generation of such pairs, we randomly skip 90% of single-object edits.
                        if (
                            random.random() < 0.9
                            or object_count_dict[str(edit_info["original_object_count"])]
                            > EXISTENCE_OBJECT_EDIT_COUNT_LIMIT
                        ):
                            continue
                        object_count_dict[str(edit_info["original_object_count"])] += 1
                    verify_already_generated_but_unfiltered_images(vqa_scorer, img_inpainted_path, edit_info)
                    was_changed = True
                logger.info(f"Skipping inpainting for: {img_inpainted_path}")

            inp_mask_path = os.path.join(output_data_dir, image_id, f"{edit_info['id']}_bbox_mask.png")
            if not os.path.exists(inp_mask_path):
                continue

            mask_image = load_image(inp_mask_path)
            mask_image_rsz = mask_image.resize((1024, 1024))

            object_name = edit_info["object_name"]
            bbox = edit_info["bbox"]

            object_count_dict[str(edit_info["original_object_count"])] += 1
            if edit_info["original_object_count"] == 1:
                # Special case: reducing original count results in object existence (zero count) image-text pair.
                # To prevent over-generation of such pairs, we randomly skip 90% of single-object edits.
                if (
                    random.random() < 0.9
                    or object_count_dict[str(edit_info["original_object_count"])] > EXISTENCE_OBJECT_EDIT_COUNT_LIMIT
                ):
                    continue

            prompt = f"{background_description}, {background_description}, realistic, blend, smooth, photo, high resolution, beautiful, dslr photo"
            logger.info(f"editing prompt: {prompt}, new caption: {edit_info['new_caption']}")
            inpainted_images = pipe(
                prompt=prompt,
                image=inp_image_rsz,
                mask_image=mask_image_rsz,
                guidance_scale=10,
                strength=0.99,
                num_images_per_prompt=5,
            ).images

            inpainted_images_rsz = [x.resize(inp_image.size) for x in inpainted_images]

            inpainted_image, tifa_result_dict = None, None
            question_answer_pairs = get_obj_removal_verify_qa_pairs_by_name(object_name)
            for imgid, image in enumerate(inpainted_images_rsz):
                segmented_image = get_segmented_image(image, edit_info.get("segmentation"))
                cropped_object = segmented_image.crop(
                    (
                        bbox[0],
                        bbox[1],
                        bbox[0] + bbox[2],
                        bbox[1] + bbox[3],
                    )
                )
                # TODO: run tifa-vqa to filter out inpainted images
                tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, cropped_object)
                inpainted_image = inpainted_images[imgid]
                if tifa_result_dict["tifa_score"] == 1:
                    break

            if tifa_result_dict["tifa_score"] < 1:
                logger.info(f"Failed to remove object via inpainting. Rejected TIFA filtering for image: {image_id}")

            inpainted_image.save(img_inpainted_path)
            edit_info.update({"new_image_path": img_inpainted_path, "tifa_score": tifa_result_dict["tifa_score"]})
            logger.info(f"tifa results: {tifa_result_dict['tifa_score']}")
            logger.info(f"saved inpainted image at: {img_inpainted_path}")
            was_changed = True

            if debug:
                og_inpainted_image_paste = np.concatenate(
                    (inp_image, inpainted_image), axis=1
                )  # paste input and inpainted image side by side
                og_inpainted_together_path = os.path.join(
                    output_data_dir, image_id, f"{edit_info['id']}_original_inpainted_side_by_side.png"
                )
                save_array_to_img(og_inpainted_image_paste, og_inpainted_together_path)
                logger.info(f"saved original and inpainted image together at: {og_inpainted_together_path}")

        if was_changed:
            # save the updated metadata
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"saved updated annotations at: {json_path}")
            was_changed = False


class CountingCaptionRefiner(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root_dir, "promptsource", f"{prompt_type}.json")

    def _prepare_prompt(
        self,
        input_dict: dict,
        num_examples_in_task_prompt: int,
    ):
        samples = self.prompt_data["samples"]

        # Sample from the selected category
        selected_examples = random.sample(samples, num_examples_in_task_prompt)
        random.shuffle(selected_examples)

        # Create the final prompt instruction with correct numbering
        task_instruction = self.prompt_data["task_instruction"] + "\n"

        order = ["Original Caption", "Removed Object", "Refined Caption"]
        sampled_examples = ""
        for i, example in enumerate(selected_examples, start=1):
            example_text = " ".join([f"{k}: {example[k]}" for k in order])
            sampled_examples += f"{i}. {example_text}\n"

        formatted_example = " ".join([f"{k}: {input_dict[k]}" for k in order if k in input_dict])
        formatted_examples = f"{sampled_examples}{formatted_example}"

        message = f"{task_instruction}{formatted_examples}"
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:  # if using Mixtral
            message = f"[INST] {message} [/INST]"

        return message


class CoutningDatagenProcessor:
    def __init__(self, llm: CountingCaptionRefiner):
        self.llm = llm

    @staticmethod
    def _load_cached_data(json_file):
        if os.path.exists(json_file):
            logger.info(f"Loading existing cache from {json_file}")
            with open(json_file, "r") as f:
                return json.load(f)

        return defaultdict(dict)

    def set_cache_file_path(self, cache_file_path):
        if not os.path.exists(os.path.dirname(cache_file_path)):
            logger.warning(f"Cache directory does not exist. Creating directory: {os.path.dirname(cache_file_path)}")
            os.makedirs(os.path.dirname(cache_file_path))
        self.cache_file_path = cache_file_path

    def refine_caption_for_counting_llm(self, data_dir, batch_size: int = 8, save_every_n: int = 50):
        image_ids = os.listdir(data_dir)
        logger.info(f"Total entries available for processing: {len(image_ids)}")
        logger.info(f"Generating enhanced phrases and caching at {self.cache_file_path}")
        self.cached_refined_data = self._load_cached_data(self.cache_file_path)
        copy_current_cache_file_as_backup_json(self.cache_file_path)

        tot_processed = 0
        image_ids_batch, prompts_batch = [], []
        num_examples_in_task_prompt = 5

        def process_batch(image_ids_batch, prompts_batch):
            start_time = time.time()
            generated_outputs = self.llm.generate_output_batch(all_prompts=prompts_batch, num_return_sequences=1)
            for image_id, prompt, output in zip(image_ids_batch, prompts_batch, generated_outputs):
                parsed_output = output.replace("Refined Caption:", "").strip()
                logger.info(f"Prompt: {prompt} => Refined Caption: {parsed_output}")
                self.cached_refined_data[image_id] = parsed_output

            end_time = time.time()
            total_time_taken = end_time - start_time
            logger.info(
                f"Total time taken to generate {len(generated_outputs)} phrases: {total_time_taken:.2f} seconds"
            )

        image_ids = [k for k in image_ids if k not in self.cached_refined_data]
        logger.info(f"Remaining entries to be processed: {len(image_ids)}")

        for image_id in tqdm(image_ids, desc="Generating conting caption", position=0, leave=True):
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            image_id = str(image_id)

            json_file_path = os.path.join(data_dir, image_id, "annotations.json")
            data = load_json_data(json_file_path)
            if data is None:
                continue

            curr_image_data = data["edit_instructions"]
            curr_image_data = [
                info for info in curr_image_data if info["original_object_count"] == 1 and info.get("tifa_score") == 1
            ]
            for info in curr_image_data:
                prompt_inp_dict = {
                    "Original Caption": data["input_caption"],
                    "Removed Object": info["object_name"],
                }
                prompt = self.llm._prepare_prompt(prompt_inp_dict, num_examples_in_task_prompt)
                image_ids_batch.append(image_id)
                prompts_batch.append(prompt)

                if len(prompts_batch) == batch_size:
                    process_batch(image_ids_batch, prompts_batch)
                    image_ids_batch, prompts_batch = [], []
                    tot_processed += 1

                if (tot_processed + 1) % save_every_n == 0:
                    save_to_json(self.cached_refined_data, self.cache_file_path)

        # Process any remaining prompts
        if prompts_batch:
            process_batch(image_ids_batch, prompts_batch)
            save_to_json(self.cached_refined_data, self.cache_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "flickr30k", "vsr"],
        help="Dataset to use. Possible values: coco, flickr30k.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use. Possible values: train, val, test.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="count_refine",
        help=f"Type of prompt to use. Possible values: count_refine. Each choice represents a different strategy for generating prompts.",
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
        choices=LANGUAGE_MODEL_NAMES,
        help=f"Set pre-trained language model to use. Possible values: {', '.join(LANGUAGE_MODEL_NAMES)}.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generating edit instructions.",
    )
    parser.add_argument(
        "--refine_caption",
        action="store_true",
        default=False,
        help="Run caption refiner to generate edit instructions.",
    )
    args = parser.parse_args()

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    dataset = args.dataset
    split = args.split
    prompt_type = args.prompt_type
    language_model_name = args.language_model_name
    debug = False
    output_data_dir = os.path.join(VIM_DATA_DIR, f"{dataset}_sdxl_removed_{split}")
    generate_edit_instructions_for_counting(split, output_data_dir)
    run_diffusion_inpainting_for_counting(split, output_data_dir, debug)

    if args.refine_caption:
        llm = CountingCaptionRefiner(
            language_model_name=language_model_name,
            prompt_type=prompt_type,
            device=device,
        )
        llm.load_pretrained_model_tokenizer()
        cnt_caption_proc = CoutningDatagenProcessor(llm)
        cache_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset}",
            "mistralaimixtral8x7binstructv0.1",
            f"counting_caption_refined_{split}.json",
        )
        cnt_caption_proc.set_cache_file_path(cache_fpath)
        cnt_caption_proc.refine_caption_for_counting_llm(output_data_dir, batch_size=args.batch_size)

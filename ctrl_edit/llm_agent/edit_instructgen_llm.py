import copy
import json
import os
import random
import re
import time
import uuid
from abc import ABC
from collections import defaultdict
from typing import List, Union

import Levenshtein
import spacy
from tqdm import tqdm

from commons.constants import SYNTH_ONLY_CATEGORIES, VALID_SPATIAL_DIRECTIONS
from commons.logger import Logger
from evals.vlm.openai_client import OpenAIGPT

from ..utils.helpers import (copy_current_cache_file_as_backup_json,
                             load_t2icompbench,
                             remove_duplicate_dict_entries_by_key)
from ..utils.llm_utils import BaseLLM

logger = Logger.get_logger(__name__)
VALID_CATEGORIES_COCO = ["object", "attribute"]


class EditInstructionGenerator(ABC):
    def __init__(
        self,
    ):
        pass

    def set_cache_file_path(self, cache_file_path):
        if not os.path.exists(cache_file_path):
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

        self.cache_file_path = cache_file_path

    def set_llm(self, llm):
        self.llm = llm

    def extract_desired_text_numbered(self, text, num_examples_in_task_prompt):
        pattern = r"\s*{}\.\s*{}:".format(
            num_examples_in_task_prompt + 2, self.llm.prompt_data["io_structure"]["input_keys"]
        )
        split_text = re.split(pattern, text)
        logger.debug(
            f"Extracting desired text from {text}, num_examples_in_task_prompt = {num_examples_in_task_prompt}, pattern = {pattern}, split_text = {split_text}"
        )
        return split_text[0].strip() if len(split_text) > 1 else text.strip()

    @staticmethod
    def extract_desired_text_multiple_newlines(text):
        text_lines = re.split(r"\n", text)
        return text_lines

    def verify_all_output_keys_exist_in_generated_output(self, generated_output):
        for key in self.llm.prompt_data["io_structure"]["output_keys"]:
            if key not in generated_output:
                return False
        return True

    @staticmethod
    def save_cache(file_path, data):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Cache saved at {file_path}")

    def _get_json_file_path(self, prompt_type: str, split: str):
        return os.path.join(self.cache_dir, f"{prompt_type}_{split}.json")

    @staticmethod
    def load_existing_annotations(json_file):
        if os.path.exists(json_file):
            logger.info(f"Loading existing cache from {json_file}")
            with open(json_file, "r") as f:
                return json.load(f)

        return defaultdict(dict)


class EditInstructgenFromCaption(EditInstructionGenerator):
    def __init__(self, cache_file_path: str, llm: Union[BaseLLM, None] = None):
        logger.info(
            f"Initializing {self.__class__.__name__} with cache_file_path: {cache_file_path}",
        )
        self.set_cache_file_path(cache_file_path)
        self.generated_annotations_dict = defaultdict(dict)
        self.llm = llm
        # Load English tokenizer, tagger, parser, NER, and word vectors
        self.nlp_model = spacy.load("en_core_web_sm")

    def is_cache_complete(self, coco_annotations: dict, num_samples_per_image: int = 5):
        self.generated_annotations_dict = self.load_existing_annotations(self.cache_file_path)

        complete_count = 0
        total_annotations = len(coco_annotations)

        for entry in coco_annotations:
            image_id, caption = entry["image_id"], entry["caption"]
            if (
                str(image_id) in self.generated_annotations_dict
                and caption in self.generated_annotations_dict[str(image_id)]
                and len(self.generated_annotations_dict[str(image_id)][caption])
                >= num_samples_per_image  # arbitrary threshold
            ):
                complete_count += 1

        if complete_count < total_annotations:
            logger.info(f"Cache completeness: {complete_count}/{total_annotations} cached.")
            return False

        return True

    def cache_full_dataset_coco_edit_instructions(
        self,
        annotations: dict,
        save_every_n: int = 100,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences=5,
    ):
        logger.info("Generating edit instructions and caching at %s", self.cache_file_path)

        start_time = time.time()

        for i, entry in enumerate(tqdm(annotations, desc="Generating edit instructions")):
            image_id, prompt = entry["image_id"], entry["caption"]

            if isinstance(image_id, int):
                image_id = str(image_id)

            if image_id in self.generated_annotations_dict and prompt in self.generated_annotations_dict[image_id]:
                logger.info(f"Skipping image id = {image_id} and prompt = {prompt} as it is already cached.")
                continue

            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            generated_output_list = self.llm.generate_output(
                input_text=prompt,
                num_return_sequences=num_return_sequences,
                num_examples_in_task_prompt=num_examples_in_task_prompt,
            )
            generated_output_list = [
                self.extract_desired_text_numbered(out, num_examples_in_task_prompt) for out in generated_output_list
            ]
            generated_output_parsed_list = [
                out for out in generated_output_list if self.verify_all_output_keys_exist_in_generated_output(out)
            ]
            logger.debug(f"Prompt: {prompt} => Generated Text: {generated_output_parsed_list}")

            # Add to cache
            if image_id not in self.generated_annotations_dict:
                self.generated_annotations_dict[image_id] = {}
            self.generated_annotations_dict[image_id][prompt] = generated_output_parsed_list

            # Occasionally save the edit instructions
            if (i + 1) % save_every_n == 0:
                self.save_cache(self.cache_file_path, self.generated_annotations_dict)
                logger.info("Saved temporary cache at iteration %d", i)

        # Save at the end
        self.save_cache(self.cache_file_path, self.generated_annotations_dict)

        elapsed_time = time.time() - start_time
        logger.info("Finished generating edit instructions in %.2f seconds", elapsed_time)

    def cache_full_dataset_coco_edit_instructions_batch(
        self,
        annotations: dict,
        dataset: str,
        max_gen_try: int = 1,
        num_samples_per_image: int = 5,
        batch_size: int = 4,
        save_every_n: int = 5,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences=10,
    ):
        logger.info(f"Generating edit instructions and caching at {self.cache_file_path}")
        self.generated_annotations_dict = self.load_existing_annotations(self.cache_file_path)
        copy_current_cache_file_as_backup_json(self.cache_file_path)

        start_time = time.time()

        prompt_batch = []
        image_ids_batch = []
        openai = OpenAIGPT()

        def process_batch_openai():
            if dataset == "vsr":
                selected_category = "object"
            else:
                selected_category = "attribute"

            for img_id, prompt in zip(image_ids_batch, prompt_batch):
                formatted_prompt = self.llm._prepare_prompt(prompt, num_examples_in_task_prompt, selected_category)
                raw_response = openai.get_response(formatted_prompt, is_json=True)
                output = raw_response.choices[0].message.content
                generated_output_parsed_list = self.parse_edit_details_from_string(output, prompt)
                print(generated_output_parsed_list)
                # make sure category matches selected category
                generated_output_parsed_list = [
                    item for item in generated_output_parsed_list if selected_category in item["category"]
                ]
                print(generated_output_parsed_list)

                img_id = str(img_id)
                if img_id not in self.generated_annotations_dict:
                    self.generated_annotations_dict[img_id] = {}

                if prompt in self.generated_annotations_dict[img_id]:
                    already_cached_list = self.generated_annotations_dict[img_id].get(prompt, [])
                    generated_output_parsed_list.extend(already_cached_list)
                    logger.info(
                        f"Extended generated_output_parsed_list by {len(already_cached_list)} items, new length: {len(generated_output_parsed_list)}"
                    )
                if generated_output_parsed_list is None:
                    continue

                for key in ["edited_caption", "edited_phrase"]:
                    generated_output_parsed_list = remove_duplicate_dict_entries_by_key(
                        generated_output_parsed_list, key
                    )

                generated_phrases = [item["edited_phrase"] for item in generated_output_parsed_list]
                logger.debug(
                    f"Prompt: {prompt} => Generated Text: {json.dumps(generated_output_parsed_list, indent=2)}"
                )
                logger.info(f"Prompt: {prompt} => Generated Phrases: {generated_phrases}")

                if not generated_phrases:  # debugging purposes
                    logger.info(f"FAILED: {output}")

                self.generated_annotations_dict[img_id][prompt] = generated_output_parsed_list

        def process_batch():
            if dataset == "vsr":
                selected_category = "object"
            else:
                # weights = [0.6, 0.4]  # Adjust these values according to your needs
                # logger.debug(f"Valid keys: {COCO_ONLY_CATEGORIES}, Weights: {weights}")
                # selected_category = random.choices(COCO_ONLY_CATEGORIES, weights=weights, k=1)[0]
                selected_category = "attribute"
            all_prompts = [
                self.llm._prepare_prompt(input_text, num_examples_in_task_prompt, selected_category)
                for input_text in prompt_batch
            ]
            flat_outputs = self.llm.generate_output_batch(
                all_prompts=all_prompts,
                max_length=512,
                num_return_sequences=num_return_sequences,
                num_examples_in_task_prompt=num_examples_in_task_prompt,
                top_k=100,
                temperature=0.9,
                stop=None,
                dataset=dataset,
            )
            # Reshape the flat_outputs
            generated_outputs_batch = [
                flat_outputs[i : i + num_return_sequences] for i in range(0, len(flat_outputs), num_return_sequences)
            ]
            for img_id, prompt, generated_output_chunk in zip(image_ids_batch, prompt_batch, generated_outputs_batch):
                generated_output_parsed_list = [
                    self.parse_edit_details_from_string(out, prompt) for out in generated_output_chunk
                ]
                # if 2d list, flatten it
                if any(isinstance(el, list) for el in generated_output_parsed_list):
                    generated_output_parsed_list = [
                        item for sublist in generated_output_parsed_list for item in sublist
                    ]

                # make sure category matches selected category
                generated_output_parsed_list = [
                    item for item in generated_output_parsed_list if selected_category in item["category"]
                ]

                img_id = str(img_id)
                if img_id not in self.generated_annotations_dict:
                    self.generated_annotations_dict[img_id] = {}

                if prompt in self.generated_annotations_dict[img_id]:
                    already_cached_list = self.generated_annotations_dict[img_id].get(prompt, [])
                    generated_output_parsed_list.extend(already_cached_list)
                    logger.info(
                        f"Extended generated_output_parsed_list by {len(already_cached_list)} items, new length: {len(generated_output_parsed_list)}"
                    )
                if generated_output_parsed_list is None:
                    continue

                for key in ["edited_caption", "edited_phrase"]:
                    generated_output_parsed_list = remove_duplicate_dict_entries_by_key(
                        generated_output_parsed_list, key
                    )

                generated_phrases = [item["edited_phrase"] for item in generated_output_parsed_list]
                logger.debug(
                    f"Prompt: {prompt} => Generated Text: {json.dumps(generated_output_parsed_list, indent=2)}"
                )
                logger.info(f"Prompt: {prompt} => Generated Phrases: {generated_phrases}")

                if not generated_phrases:  # debugging purposes
                    logger.info(f"FAILED: {generated_outputs_batch[0]}")

                self.generated_annotations_dict[img_id][prompt] = generated_output_parsed_list

        for i, entry in enumerate(tqdm(annotations, desc="Generating edit instructions")):
            image_id, prompt = entry["image_id"], entry["caption"]

            if (
                str(image_id) in self.generated_annotations_dict
                and prompt in self.generated_annotations_dict[str(image_id)]
                and len(self.generated_annotations_dict[str(image_id)][prompt])
                >= num_samples_per_image  # arbitrary threshold
            ):
                logger.debug(f"Skipping image id = {image_id}")
                continue

            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            prompt_batch.append(prompt)
            image_ids_batch.append(image_id)

            if len(prompt_batch) == batch_size:
                # Process the batch run for the max try count and then clear the batch
                for _ in range(max_gen_try):
                    process_batch()
                # process_batch_openai()
                prompt_batch = []
                image_ids_batch = []
                # Occasionally save the edit instructions
                if (i + 1) % save_every_n == 0:
                    self.save_cache(self.cache_file_path, self.generated_annotations_dict)
                    logger.info(f"Saved temporary cache at iteration {i}")
                    copy_current_cache_file_as_backup_json(self.cache_file_path)

        # Process remaining items in the batch
        if prompt_batch:
            for _ in range(max_gen_try):
                process_batch()
            # process_batch_openai()

        # Save at the end
        self.save_cache(self.cache_file_path, self.generated_annotations_dict)
        copy_current_cache_file_as_backup_json(self.cache_file_path)

        elapsed_time = time.time() - start_time
        logger.info("Finished generating edit instructions in %.2f seconds", elapsed_time)

    def find_noun_phrases(self, sentence, target_phrase):
        processed_sentence = sentence.lower()
        doc = self.nlp_model(processed_sentence)

        for chunk in doc.noun_chunks:
            if target_phrase.lower() in chunk.text.lower():
                return chunk.text

            if chunk.text.lower() in target_phrase.lower() and target_phrase.lower() in processed_sentence:
                return target_phrase

            if target_phrase.lower() in processed_sentence:
                return target_phrase

        return None

    def parse_edit_details_from_string(self, input_str: str, prompt: str):
        logger.debug(f"INPUT STRING: {input_str}")
        edit_pattern = r'\{\s*"InputCaption": "(.*?)",\s*"SelectedPhrase": "(.*?)",\s*"EditedPhrase": "(.*?)",\s*"EditedRegionPhrase": "(.*?)",\s*"EditedCaption": "(.*?)",\s*"Category": "(.*?)"\s*\}'
        matches = re.findall(edit_pattern, input_str)

        extracted_data = []
        for match in matches:
            logger.debug(f"Match: {match}")
            try:
                input_text = match[0].strip()
                if Levenshtein.distance(input_text, prompt) > 5:
                    logger.info(f"mis-match: {input_text} != {prompt}")
                    continue
                selected_phrase = match[1].strip()
                edited_phrase = match[2].strip()
                edited_region_phrase = match[3].strip()
                edited_caption = match[4].strip()
                category = match[5].strip()

                if Levenshtein.distance(edited_phrase.lower(), "no attribute change") < 2:
                    continue
                # if the category does not contain 'object', 'attribute', skip this
                if not any(cond in category.split("(")[0] for cond in VALID_CATEGORIES_COCO):
                    continue

                # Adjust the parsing logic as needed
                question_data = {
                    "input_phrase": selected_phrase,
                    "edited_phrase": edited_phrase,
                    "edited_caption": edited_caption,
                    "edited_region_phrase": edited_region_phrase,
                    "category": category,
                    "edit_id": str(uuid.uuid4())[:8],
                }
                extracted_data.append(question_data)
            except Exception as e:
                logger.error(f"Error extracting data: {e}")
                continue

        return extracted_data


class EditInstructgenBootstrap(EditInstructionGenerator):
    def __init__(self, cache_file_path: str, llm: Union[BaseLLM, None] = None, max_num_samples: int = 10000):
        logger.info(f"Initializing {self.__class__.__name__} with cache_file_path: {cache_file_path}")
        self.set_cache_file_path(cache_file_path)
        self.generated_annotations_dict = defaultdict(dict)
        self.llm = llm
        self.max_edit_instructions = max_num_samples

    def is_cache_complete(self):
        self.generated_annotations_dict = self.load_existing_annotations(self.cache_file_path)

        cached_image_ids = set(self.generated_annotations_dict.keys())
        cached_count = len(cached_image_ids)

        logger.info(f"Cache completeness: {cached_count}/{self.max_edit_instructions} cached.")
        if cached_count < self.max_edit_instructions:
            return False

        return True

    def get_unique_captions(self):
        unique_captions = set()
        for generated_output in self.generated_annotations_dict.values():
            unique_captions.add(generated_output["prompt"])
        return unique_captions

    def extend_promptsource_with_t2icompbench(self):
        logger.info("Extending promptsource with t2icompbench")
        t2icompbench = load_t2icompbench()

        for i in range(0, len(t2icompbench), 10):
            batch = t2icompbench[i : i + 10]
            generated_output_list = self.llm.generate_output_batch(
                input_texts=batch,
                num_return_sequences=5,
                num_examples_in_task_prompt=8,
                max_length=640,
                stop=None,
            )
            generated_output_list = [self.parse_layout_details_from_string(out) for out in generated_output_list]
            if any(isinstance(el, list) for el in generated_output_list):
                generated_output_list = [item for sublist in generated_output_list for item in sublist]

    def cache_full_dataset_bootstrap_edit_instructions(
        self,
        dataset,
        save_every_n: int = 10,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences=5,
        batch_size: int = 10,
        batch_generate=False,
    ):
        logger.info(f"Generating edit instructions and caching at {self.cache_file_path}")
        self.generated_annotations_dict = self.load_existing_annotations(self.cache_file_path)

        unique_caption_set = self.get_unique_captions()
        required_num_samples = max(self.max_edit_instructions - len(self.generated_annotations_dict), 0) // (
            batch_size * num_return_sequences
        )
        for i in tqdm(range(required_num_samples), desc="Generating edit instructions"):
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            start_time = time.time()
            prompt = ""  # set prompt to empty string for bootstrap

            if batch_generate:
                prompts = [prompt] * batch_size
                generated_output_list = self.llm.generate_output_batch(
                    input_texts=prompts,
                    num_return_sequences=num_return_sequences,
                    num_examples_in_task_prompt=num_examples_in_task_prompt,
                    max_length=640,
                    stop=None,
                    dataset=dataset,
                )
            else:
                generated_output_list = self.llm.generate_output(
                    input_text=prompt,
                    num_return_sequences=num_return_sequences,
                    num_examples_in_task_prompt=num_examples_in_task_prompt,
                    max_length=512,
                )
            generated_output_list = [self.parse_layout_details_from_string(out) for out in generated_output_list]
            if any(isinstance(el, list) for el in generated_output_list):  # 2d list, flatten it
                generated_output_list = [item for sublist in generated_output_list for item in sublist]

            if not generated_output_list:
                continue

            # Add to cache
            valid_generated_output_list = []
            for generated_output in generated_output_list:
                if generated_output["prompt"] in unique_caption_set:
                    continue
                image_id = str(uuid.uuid4())[:8]
                self.generated_annotations_dict[image_id] = generated_output
                unique_caption_set.add(generated_output["prompt"])
                valid_generated_output_list.append(generated_output)

            logger.info(f"Prompt: {prompt} => Generated Text: {valid_generated_output_list}")

            elapsed_time = time.time() - start_time
            logger.info(
                f"Time taken to generate {len(generated_output_list)} edit instructions: {elapsed_time:.2f} seconds"
            )

            # Occasionally save the edit instructions
            if (i + 1) % save_every_n == 0:
                self.save_cache(self.cache_file_path, self.generated_annotations_dict)
                logger.info("Saved temporary cache at iteration %d", i)

        # Save at the end
        self.save_cache(self.cache_file_path, self.generated_annotations_dict)

    def detect_bboxes_with_large_area_difference(self, bounding_box_data: list):
        # check the area of first object and second object, if the difference is more than 50% then skip,
        # the values are not bounded by 0 and 1
        area_first = bounding_box_data[0][1][2] * bounding_box_data[0][1][3]
        area_second = bounding_box_data[1][1][2] * bounding_box_data[1][1][3]
        if abs(area_first - area_second) / max(area_first, area_second) > 0.5:
            return True

        return False

    def detect_overlapping_bboxes(self, bounding_box_data: list):
        for i in range(len(bounding_box_data)):
            for j in range(i + 1, len(bounding_box_data)):
                if self.check_overlap(bounding_box_data[i][1], bounding_box_data[j][1]):
                    return True
        return False

    # check if the bounding boxes are overlapping
    def check_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
            return True
        return False

    @staticmethod
    def add_indefinite_article_to_object_names(bounding_box_data: list):
        bounding_box_data_copy = copy.deepcopy(bounding_box_data)

        for i in range(len(bounding_box_data_copy)):
            obj_name = bounding_box_data_copy[i][0]
            obj_name = obj_name.strip()

            if not obj_name.split()[0].lower() in ["a", "an"]:
                # Use a regular expression to check if the first character is a vowel sound for 'an'
                if re.match(r"^(?:[aeiou]|hon|hour|herb)", obj_name, re.I):
                    bounding_box_data_copy[i][0] = f"an {obj_name}"
                else:
                    bounding_box_data_copy[i][0] = f"a {obj_name}"

        return bounding_box_data_copy

    def parse_layout_details_from_string(self, input_str):
        pattern = (
            r"(?:\d+\.\s)?Input: (.*?), Bounding boxes: (\[.*?\]),"
            r" Background prompt: (.*?), Negative prompt: (.*?),"
            r" Category: (.*?)(?=(?:\d+\.\s)?Input:|$)"
        )

        matches = re.finditer(pattern, input_str, re.DOTALL)
        extracted_data = []
        for match in matches:
            try:
                input_description = match.group(1).strip()
                bounding_boxes = match.group(2).strip()
                bounding_boxes = eval(bounding_boxes)
                background_prompt = match.group(3).strip()
                negative_prompt = match.group(4).strip() if match.group(4) else ""
                category_subcategory = match.group(5).strip()
                category = category_subcategory.split("(")[0]
                print(input_description, bounding_boxes, background_prompt, negative_prompt, category_subcategory)
                # make sure the category is one of the valid categories
                if not any(cond in category for cond in SYNTH_ONLY_CATEGORIES):
                    continue

                # directions are valid and the bounding boxes are not too different in area for relation category
                if category == "relation":
                    if all(direction not in input_description for direction in VALID_SPATIAL_DIRECTIONS):
                        continue
                    if len(bounding_boxes) != 2 or self.detect_bboxes_with_large_area_difference(bounding_boxes):
                        continue

                # check overlapping bounding boxes, more than one bounding boxes for counting category
                if category == "counting":
                    if self.detect_overlapping_bboxes(bounding_boxes):
                        continue
                    if len(bounding_boxes) < 2:
                        continue

                bounding_boxes = self.add_indefinite_article_to_object_names(bounding_boxes)
                question_data = {
                    "prompt": input_description,
                    "bounding_boxes": bounding_boxes,
                    "background_prompt": background_prompt,
                    "negative_prompt": negative_prompt,
                    "category": category_subcategory,
                }
                extracted_data.append(question_data)
            except Exception as e:
                print(f"Error extracting data, error: {e}")
                continue

        return extracted_data

import argparse
import ast
import json
import os
import random
import re
import string
import time
from collections import defaultdict
from itertools import islice
from typing import List

import torch
from tqdm import tqdm

from commons.constants import (LANGUAGE_MODEL_NAMES,
                               MISTRALAI_LANGUAGE_MODEL_NAMES,
                               SYNTH_DIFFUSE_DATA_DIR, TOTAL_NUM_COCO_CHUNKS,
                               VALID_SPATIAL_DIRECTIONS)
from commons.logger import Logger
from tifa.tifascore import UnifiedQAModel, filter_question_and_answers

from ..utils.helpers import (copy_current_cache_file_as_backup_json,
                             group_outputs_for_batch_repeats, load_json_data,
                             remove_current_cache_backup_file, save_to_json,
                             set_random_seed)
from ..utils.llm_utils import BaseLLM, save_tifa_prompt_demo_data
from ..utils.spatial_relation_utils import \
    extract_spatial_direction_from_caption
from ..utils.vqa_utils import (OBJECT_ABSENT_PROMPT, OBJECT_COUNT_PROMPT,
                               OBJECT_EXIST_ABSENT_PROMPT)

logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()


class GenerateQuestionsForEditedImage(BaseLLM):
    QUESTION_PATTERN = re.compile(r'Question: (.*?) Choices: (\[.*?\]) Answer: (.*?)(?=", "Question|$)')

    def __init__(self, language_model_name: str, dataset: str, prompt_type: str, device: str):
        output_filepath = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", "tifa_qa_prompts.json")
        save_tifa_prompt_demo_data(output_filepath)
        super().__init__(language_model_name, prompt_type, device)
        self.dataset = dataset
        self.cached_generation_dict = defaultdict(dict)
        self.unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")

    def set_cache_file_path(self, cache_file_path):
        if not os.path.exists(os.path.dirname(cache_file_path)):
            logger.warning(f"Cache directory does not exist. Creating directory: {os.path.dirname(cache_file_path)}")
            os.makedirs(os.path.dirname(cache_file_path))
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

    def _prepare_prompt(self, image_caption: str, num_examples_in_task_prompt: int, caption_type: str):
        examples = self.prompt_data["samples"]
        examples = [example for example in examples if len(example.get("Output", "")) > 3]
        random.shuffle(examples)
        LIMIT = {"full": 2, "region": 6, "region_obj_count": 8, "region_neg_only": 6}.get(caption_type)
        selected_examples = random.sample(examples, num_examples_in_task_prompt - LIMIT)

        if caption_type == "region_neg_only":
            selected_examples += random.sample(OBJECT_ABSENT_PROMPT, LIMIT)
        elif caption_type == "region_obj_count":
            selected_examples += random.sample(OBJECT_COUNT_PROMPT, LIMIT)
        else:
            selected_examples += random.sample(OBJECT_EXIST_ABSENT_PROMPT, LIMIT)
        random.shuffle(selected_examples)

        # capitalize the first letter of the caption and add a period at the end if it doesn't exist
        if image_caption:
            image_caption = image_caption[0].upper() + image_caption[1:]
        if not image_caption.endswith("."):
            image_caption += "."

        if caption_type == "region":
            prompt_instruction = self.prompt_data["task_instruction_region_caption"]
        elif caption_type == "region_obj_count":
            prompt_instruction = self.prompt_data["task_instruction_region_obj_count_caption"]
        elif caption_type == "region_neg_only":
            prompt_instruction = self.prompt_data["task_instruction_region_neg_only_caption"]
        else:
            prompt_instruction = self.prompt_data["task_instruction_full_caption"]

        for i, example in enumerate(selected_examples, start=1):
            if isinstance(example, dict):
                example = " ".join([f"{k}: {v}" for k, v in example.items()])
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{{input_key}}: {{image_caption}} {{output_key}}:"

        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            image_caption=image_caption,
            output_key=self.prompt_data["io_structure"]["output_keys"],
        )
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:
            prompt = "[INST]" + prompt + "[/INST]"
        return prompt

    def generate_questions_from_caption(
        self,
        input_texts: List[str],
        caption_type: str,
        max_length: int = 512,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences: int = 1,
    ) -> str:
        if caption_type not in ["full", "region", "region_obj_count", "region_neg_only"]:
            raise ValueError(f"Invalid caption type: {caption_type}")
        all_prompts = [self._prepare_prompt(text, num_examples_in_task_prompt, caption_type) for text in input_texts]
        logger.debug(f"all prompts: {json.dumps(all_prompts, indent=2)}")

        # Generate unique r_temp and r_top_k for this batch
        r_temp = random.uniform(0.7, 0.9)
        r_top_k = random.randint(40, 70)
        generated_outputs = self.generate_output_batch(
            all_prompts=all_prompts,
            max_length=max_length,
            num_examples_in_task_prompt=num_examples_in_task_prompt,
            num_return_sequences=num_return_sequences,
            temperature=r_temp,
            top_k=r_top_k,
        )
        logger.debug(f"Generated outputs: {json.dumps(generated_outputs, indent=2)}")
        return generated_outputs

    def process_batch_edit_ann(self, image_ids_batch, uuids_batch, caption_info_batch):
        start_time = time.time()
        logger.debug(f"{json.dumps(caption_info_batch, indent=4)}")

        # !!! IMPORTANT HACK !!!
        # WARNING: we're repeating the prompt for each example to increase diversity in the task prompt
        REPEAT_PROMPT_TIMES = 2
        num_return_sequences = 2  # kinda arbitrary, but it's for retrying if the first set is not good enough
        all_generated_outputs_fullcap, all_generated_outputs_regcap, all_generated_output_inpnegcap = [], [], []

        (
            caption_p_region_texts_batch,
            caption_p_edited_phrase_texts_batch,
            region_caption_texts_batch,
            inp_phrase_neg_texts_batch,
        ) = ([], [], [], [])
        for caption_info in caption_info_batch:
            edited_caption = caption_info["edited_caption"]
            input_phrase = caption_info["input_phrase"][0].lower() + caption_info["input_phrase"][1:]
            input_phrase += "." if not input_phrase.endswith(".") else ""
            neg_input_phrase = f"Not {input_phrase}"
            edited_phrase = caption_info["edited_phrase"][0].upper() + caption_info["edited_phrase"][1:]
            edited_phrase += "." if not edited_phrase.endswith(".") else ""
            region_caption = f"{edited_phrase} {neg_input_phrase}"
            caption_p_region_text = f"{region_caption} {edited_caption}"
            caption_p_edited_phrase_text = f"{edited_phrase} {edited_caption}"

            caption_p_region_texts_batch.append(caption_p_region_text)
            caption_p_edited_phrase_texts_batch.append(caption_p_edited_phrase_text)
            region_caption_texts_batch.append(region_caption)
            inp_phrase_neg_texts_batch.append(neg_input_phrase)

        for _ in range(REPEAT_PROMPT_TIMES):
            generated_questions = self.generate_questions_from_caption(
                caption_p_region_texts_batch, "full", num_return_sequences=num_return_sequences + 1
            )
            all_generated_outputs_fullcap.extend(generated_questions)

            generated_questions = self.generate_questions_from_caption(
                region_caption_texts_batch, "region", num_return_sequences=num_return_sequences
            )
            all_generated_outputs_regcap.extend(generated_questions)

        for _ in range(1):
            generated_questions = self.generate_questions_from_caption(
                inp_phrase_neg_texts_batch, "region_neg_only", num_return_sequences=num_return_sequences
            )
            all_generated_output_inpnegcap.extend(generated_questions)

        # chunk the generated questions into lists of num_return_sequences*REPEAT_PROMPT_TIMES
        grouped_outputs_fullcap = group_outputs_for_batch_repeats(
            all_generated_outputs_fullcap, batch_size, REPEAT_PROMPT_TIMES, num_return_sequences + 1
        )
        grouped_outputs_regcap = group_outputs_for_batch_repeats(
            all_generated_outputs_regcap, batch_size, REPEAT_PROMPT_TIMES, num_return_sequences
        )
        grouped_outputs_inpnegcap = group_outputs_for_batch_repeats(
            all_generated_output_inpnegcap, batch_size, 1, num_return_sequences
        )
        logger.debug(f"Generated questions: {json.dumps(grouped_outputs_fullcap, indent=4)}")

        if len(grouped_outputs_fullcap) != len(caption_p_region_texts_batch):
            logger.error(
                f"Mismatch in count: generated questions ({len(grouped_outputs_fullcap)}) vs input captions ({len(caption_p_region_texts_batch)}). They should be equal."
            )
            return

        if len(grouped_outputs_regcap) != len(region_caption_texts_batch):
            logger.error(
                f"Mismatch in count: generated questions ({len(grouped_outputs_regcap)}) vs input captions ({len(region_caption_texts_batch)}). They should be equal."
            )
            return

        # TODO: test why so many questions are being rejected (less_than_threshold_qa_pairs_count)
        less_than_threshold_qa_pairs_count = 0
        for (
            image_id,
            edit_id,
            gen_qa_fullcap_chunk,
            gen_qa_regcap_chunk,
            gen_qa_inpnegcap_chunk,
            caption_p_region_text,
            caption_p_edited_phrase_text,
        ) in zip(
            image_ids_batch,
            uuids_batch,
            grouped_outputs_fullcap,
            grouped_outputs_regcap,
            grouped_outputs_inpnegcap,
            caption_p_region_texts_batch,
            caption_p_edited_phrase_texts_batch,
        ):
            if isinstance(gen_qa_fullcap_chunk, str):
                gen_qa_fullcap_chunk = [gen_qa_fullcap_chunk]

            parsed_qa_pairs_fullcap = [
                self.parse_qa_pairs_from_string(gen_qa_str) for gen_qa_str in gen_qa_fullcap_chunk
            ]
            parsed_qa_pairs_regcap = [self.parse_qa_pairs_from_string(gen_qa_str) for gen_qa_str in gen_qa_regcap_chunk]
            parsed_qa_pairs_inpnegcap = [
                self.parse_qa_pairs_from_string(gen_qa_str) for gen_qa_str in gen_qa_inpnegcap_chunk
            ]

            if any(isinstance(el, list) for el in parsed_qa_pairs_fullcap):  # 2d list, flatten it
                parsed_qa_pairs_fullcap = [item for sublist in parsed_qa_pairs_fullcap for item in sublist]

            if any(isinstance(el, list) for el in parsed_qa_pairs_regcap):  # 2d list, flatten it
                parsed_qa_pairs_regcap = [item for sublist in parsed_qa_pairs_regcap for item in sublist]

            if any(isinstance(el, list) for el in parsed_qa_pairs_inpnegcap):  # 2d list, flatten it
                parsed_qa_pairs_inpnegcap = [item for sublist in parsed_qa_pairs_inpnegcap for item in sublist]

            parsed_qa_pairs_fullcap = filter_question_and_answers(
                self.unifiedqa_model, parsed_qa_pairs_fullcap, caption_p_edited_phrase_text
            )
            parsed_qa_pairs_regcap = filter_question_and_answers(
                self.unifiedqa_model, parsed_qa_pairs_regcap, caption_p_edited_phrase_text
            )
            parsed_qa_pairs_inpnegcap = filter_question_and_answers(
                self.unifiedqa_model, parsed_qa_pairs_inpnegcap, caption_p_region_text
            )

            if (
                not parsed_qa_pairs_fullcap
                or len(parsed_qa_pairs_fullcap) < 3
                or not parsed_qa_pairs_regcap
                or len(parsed_qa_pairs_regcap) < 2
                or not parsed_qa_pairs_inpnegcap
                or len(parsed_qa_pairs_inpnegcap) < 1
            ):
                logger.debug(parsed_qa_pairs_fullcap)
                logger.debug(parsed_qa_pairs_regcap)
                less_than_threshold_qa_pairs_count += 1
                continue
            self.cached_generation_dict[image_id][edit_id] = {
                "generated_questions_full_caption": parsed_qa_pairs_fullcap,
                "generated_questions_region_caption": parsed_qa_pairs_regcap + parsed_qa_pairs_inpnegcap,
                "prompt": caption_p_region_text,
            }
            logger.debug(f"{caption_p_region_text} => {parsed_qa_pairs_fullcap}")
            logger.info(f"{caption_p_region_text} => {parsed_qa_pairs_regcap+parsed_qa_pairs_inpnegcap}")

        end_time = time.time()
        total_time_taken = end_time - start_time
        logger.info(
            f"Total time taken to generate qa pairs for {len(grouped_outputs_fullcap)} batch size: {total_time_taken:.2f} seconds"
        )
        logger.info(
            f"Less than threshold (in generated qa pairs) count: {less_than_threshold_qa_pairs_count} out of {len(grouped_outputs_fullcap)}"
        )

    def load_cached_llm_filtered_edits_by_chunk_index(self, dataset: str, split: str, chunk_index: int):
        if chunk_index is not None:
            filepath = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR,
                "prompt_resources",
                f"llm_edits_{dataset}",
                "mistralaimixtral8x7binstructv0.1",
                f"edit_instruction_filtered_{split}_chunked",
                f"chunk_{chunk_index}.json",
            )
        else:
            filepath = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR,
                "prompt_resources",
                f"llm_edits_{dataset}",
                "mistralaimixtral8x7binstructv0.1",
                f"edit_instruction_filtered_{split}.json",
            )
        try:
            logger.info(f"Loading cached llm filtered edit instructions from {filepath}")
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find cached edit instructions file at {filepath}")

    def generate_qa_from_edit_annotations(
        self, annotations: dict, llm_filtered_edits: dict, batch_size: int = 4, save_every_n: int = 10
    ):
        logger.info(f"Total entries available for processing: {len(annotations)}")
        logger.info(f"Generating question-answer pairs and caching at {self.cache_file_path}")
        self.cached_generation_dict = self._load_cached_data(self.cache_file_path)
        copy_current_cache_file_as_backup_json(self.cache_file_path)
        image_ids_batch, uuids_batch, caption_info_batch = [], [], []

        # Keep the key-value pair in annotations if the key k does not exist in self.cached_generation_dict
        # or if the lengths of the values corresponding to k in annotations and self.cached_generation_dict are not equal.
        annotations = {
            k: v
            for k, v in annotations.items()
            if k not in self.cached_generation_dict
            or len(annotations.get(k, {})) != len(self.cached_generation_dict[k])
        }
        logger.info(f"Remaining entries to be processed: {len(annotations)}")

        update_cache = False
        for i, image_id in enumerate(tqdm(annotations, desc="Generating questions")):
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")
            image_id = str(image_id)
            curr_image_data = annotations[image_id]

            if image_id not in self.cached_generation_dict:
                self.cached_generation_dict[image_id] = {}

            for curr_list_of_dicts in curr_image_data.values():
                # need those which are not already in the cache and are not rejected
                curr_llm_filtered_data = llm_filtered_edits.get(str(image_id), {})
                filtered_image_edit_ids = []
                for edit_id, edit_info in curr_llm_filtered_data.items():
                    if edit_info["reject"] == "NO":
                        filtered_image_edit_ids.append(edit_id)
                all_images_edit_ids = [info["edit_id"] for info in curr_list_of_dicts]
                curr_list_of_dicts = [
                    info
                    for info in curr_list_of_dicts
                    if info["edit_id"] in filtered_image_edit_ids
                    and info["edit_id"] not in self.cached_generation_dict[image_id]
                ]
                logger.info(
                    f"Skipped {len(all_images_edit_ids) - len(curr_list_of_dicts)} cached edits out of {len(all_images_edit_ids)}"
                )

                for info in curr_list_of_dicts:
                    edit_id = info["edit_id"]
                    # I suppose I wanted to support both DINO-based edits (localized) vs generated from scratch (bootstrap approach)
                    # caption_key_name = "edited_phrase" if "dino" in data_dir else "source_caption"
                    uuids_batch.append(edit_id)
                    if len(info["edited_phrase"]) < 3:
                        continue
                    image_ids_batch.append(image_id)
                    caption_info_batch.append(info)

                    if len(caption_info_batch) == batch_size:
                        self.process_batch_edit_ann(image_ids_batch, uuids_batch, caption_info_batch)
                        image_ids_batch, uuids_batch, caption_info_batch = ([], [], [])
                        update_cache = True

            if (i + 1) % save_every_n == 0 and update_cache:
                save_to_json(self.cached_generation_dict, self.cache_file_path)
                copy_current_cache_file_as_backup_json(self.cache_file_path)
                update_cache = False

        if caption_info_batch:
            self.process_batch_edit_ann(image_ids_batch, uuids_batch, caption_info_batch)
            save_to_json(self.cached_generation_dict, self.cache_file_path)

        remove_current_cache_backup_file(self.cache_file_path)

    def process_batch_layout_ann(self, image_ids_batch, bbox_info_batch):
        input_texts_batch = []
        if self.dataset == "relation":
            for bounding_box_data in bbox_info_batch:
                input_texts = [f"{bounding_box_data[i][0]}. Not {bounding_box_data[1-i][0]}" for i in range(2)]
                input_texts_batch.append(input_texts)
        else:
            for bounding_box_data in bbox_info_batch:
                input_texts = [f"{curr_bbox_data[0]}" for curr_bbox_data in bounding_box_data]
                input_texts = list(set(input_texts))
                input_texts_batch.append(input_texts)

        input_texts_flat = [item for sublist in input_texts_batch for item in sublist]
        if self.dataset == "counting":
            input_texts_flat = [f"A photo of a {text}." for text in input_texts_flat]
        input_texts_flat = [
            text[0].upper() + text[1:] + ("." if not text.endswith(".") else "") for text in input_texts_flat
        ]

        REPEAT_PROMPT_TIMES = 2
        num_return_sequences = 3
        caption_type = "region" if self.dataset == "relation" else "region_obj_count"
        all_generated_outputs = []
        for _ in range(REPEAT_PROMPT_TIMES):
            generated_questions = self.generate_questions_from_caption(
                input_texts_flat, caption_type=caption_type, num_return_sequences=num_return_sequences
            )
            all_generated_outputs.extend(generated_questions)

        generated_outputs_chunked = group_outputs_for_batch_repeats(
            all_generated_outputs, len(input_texts_flat), REPEAT_PROMPT_TIMES, num_return_sequences
        )

        less_than_threshold_qa_pairs_count = 0
        for image_id, input_texts, bounding_box_data in zip(image_ids_batch, input_texts_batch, bbox_info_batch):
            generated_questions_filt_chunk = {}
            for input_text in input_texts:
                parsed_qa_pairs = [
                    self.parse_qa_pairs_from_string(gen_qa_str) for gen_qa_str in generated_outputs_chunked.pop(0)
                ]
                if any(isinstance(el, list) for el in parsed_qa_pairs):  # 2d list, flatten it
                    parsed_qa_pairs = [item for sublist in parsed_qa_pairs for item in sublist]
                parsed_qa_pairs = filter_question_and_answers(self.unifiedqa_model, parsed_qa_pairs, input_text)
                if not parsed_qa_pairs or len(parsed_qa_pairs) < 4:
                    continue
                generated_questions_filt_chunk[input_text] = parsed_qa_pairs
                logger.info(f"input_text: {input_text} => {parsed_qa_pairs} [END]")

            if len(generated_questions_filt_chunk) == len(input_texts):
                self.cached_generation_dict[image_id] = {
                    "generated_questions": list(generated_questions_filt_chunk.values())
                    if self.dataset == "relation"
                    else generated_questions_filt_chunk
                }
            else:
                less_than_threshold_qa_pairs_count += 1

        logger.info(
            f"Less than threshold (in generated qa pairs) count: {less_than_threshold_qa_pairs_count} out of {len(image_ids_batch)}"
        )

    def load_annotated_counting_image_ids(self, split: str):
        if split == "train":
            split = "training"
        fpath = f"/network/projects/mair_project_vim/annotations/{split}_data/counting.jsonl"
        with open(fpath, "r") as f:
            data = f.readlines()
        data = [json.loads(item) for item in data]
        image_ids = [item["image_id"] for item in data]
        return image_ids

    def generate_qa_from_llm_layout_annotations(self, annotations: dict, batch_size: int, save_every_n=10):
        # if self.dataset == "counting":
        #     annotated_image_ids = self.load_annotated_counting_image_ids("validation")

        logger.info(f"Total entries available for processing: {len(annotations)}")
        logger.info(f"Generating question-answer pairs and caching at {self.cache_file_path}")
        self.cached_generation_dict = self._load_cached_data(self.cache_file_path)

        update_cache = False
        image_ids_batch, bbox_info_batch = [], []
        for i, image_id in enumerate(tqdm(annotations, desc="Generating questions")):
            # if self.dataset == "counting" and image_id not in annotated_image_ids:
            #     print(f"Ignoring image_id: {image_id} as it is not annotated for Mturk.")
            #     continue
            if image_id in self.cached_generation_dict:
                continue

            curr_image_data = annotations[image_id]
            bounding_box_data = curr_image_data["bounding_boxes"]
            if isinstance(bounding_box_data, str):
                bounding_box_data = eval(bounding_box_data)

            if self.dataset == "relation":
                if len(bounding_box_data) != 2:
                    continue
                if extract_spatial_direction_from_caption(curr_image_data["prompt"]) is None:
                    continue
            print(f"image_id: {image_id}, caption: {annotations[image_id]['prompt']}")
            image_ids_batch.append(image_id)
            bbox_info_batch.append(bounding_box_data)

            if len(image_ids_batch) == batch_size:
                self.process_batch_layout_ann(image_ids_batch, bbox_info_batch)
                image_ids_batch, bbox_info_batch = [], []
                update_cache = True

            if (i + 1) % save_every_n == 0 and update_cache:
                save_to_json(self.cached_generation_dict, self.cache_file_path)
                update_cache = False

        if image_ids_batch:
            self.process_batch_layout_ann(image_ids_batch, bbox_info_batch)
            save_to_json(self.cached_generation_dict, self.cache_file_path)

    def parse_qa_pairs_from_string(self, input_string) -> list:
        """
        Extracts questions, choices, and answers from the provided string.

        Args:
        - input_string (str): The string containing questions, choices, and answers.

        Returns:
        - list of dict: Each dictionary contains a Question, its Choices, and the Answer.
        """
        # question_pattern = r'Question: (.*?) Choices: (\[.*?\]) Answer: (.*?)(?=", "Question|$)'
        matches = re.findall(self.QUESTION_PATTERN, input_string)

        extracted_data = []
        for match in matches:
            try:
                question_text = match[0].strip()
                choice_string = match[1]
                choice_list = ast.literal_eval(choice_string)
                provided_answer = match[2].translate(str.maketrans("", "", string.punctuation)).strip()

                question_data = {
                    "question": question_text,
                    "choices": choice_list,
                    "answer": provided_answer,
                }
                extracted_data.append(question_data)
            except Exception as e:
                logger.error(f"Error extracting question data: {e}")
                continue

        return extracted_data


# This script is now divided into 15 chunks, with multiple jobs each handling a unique set of images.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question generation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "vsr", "counting", "relation"],
        help="Dataset to use. Possible values: coco, flickr30k.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, val, test.",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        help=f"Index of the chunk to process. Possible values: 0-{TOTAL_NUM_COCO_CHUNKS-1}.",
    )
    parser.add_argument(
        "--language_model_name",
        default="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
        choices=LANGUAGE_MODEL_NAMES,
        help="Language model name",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    args = parser.parse_args()

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    dataset = args.dataset
    split = args.split
    chunk_index = args.chunk_index
    batch_size = args.batch_size
    language_model_name = args.language_model_name
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()

    if dataset == "coco" and split == "train" and chunk_index is None:
        raise ValueError("Please provide a chunk index for the COCO train split.")

    src_prefix = "edit_suggestion" if dataset in ["coco", "vsr"] else "llm_layout"
    if chunk_index is not None:
        input_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset}",
            "mistralaimixtral8x7binstructv0.1",
            f"{src_prefix}_{split}_chunked",
            f"chunk_{chunk_index}.json",
        )
        output_file_path = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"qa_annotations_{dataset}_{split}_chunked",
            f"chunk_{chunk_index}.json",
        )
    else:
        input_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset}",
            "mistralaimixtral8x7binstructv0.1",
            f"{src_prefix}_{split}.json",
        )
        output_file_path = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"qa_annotations_{dataset}_{split}.json",
        )

    qa_gen_engine = GenerateQuestionsForEditedImage(
        language_model_name=language_model_name,
        dataset=dataset,
        prompt_type="tifa_qa_prompts",
        device="cuda",
    )
    qa_gen_engine.set_cache_file_path(output_file_path)
    annotations = load_json_data(input_fpath)
    qa_gen_engine.load_pretrained_model_tokenizer()
    if dataset in ["coco", "vsr"]:
        llm_filtered_edits = qa_gen_engine.load_cached_llm_filtered_edits_by_chunk_index(dataset, split, chunk_index)
        qa_gen_engine.generate_qa_from_edit_annotations(annotations, llm_filtered_edits, batch_size)
    elif dataset in ["relation", "counting"]:
        qa_gen_engine.generate_qa_from_llm_layout_annotations(annotations, batch_size)

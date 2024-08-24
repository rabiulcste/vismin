import argparse
import ast
import json
import os
import random
import re
import time
from collections import defaultdict
from itertools import islice

import torch
from tqdm import tqdm

from commons.constants import (LANGUAGE_MODEL_NAMES,
                               MISTRALAI_LANGUAGE_MODEL_NAMES,
                               SYNTH_DIFFUSE_DATA_DIR, TOTAL_NUM_COCO_CHUNKS,
                               VALID_CATEGORY_NAMES)
from commons.logger import Logger

from ..utils.helpers import (copy_current_cache_file_as_backup_json,
                             remove_current_cache_backup_file, save_to_json,
                             set_random_seed)
from ..utils.llm_utils import BaseLLM

# Set up logger
logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()


class PromptEnhancerLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root_dir, "promptsource", f"{prompt_type}.json")

    def _prepare_prompt(
        self,
        phrase: str,
        image_caption: str,
        num_examples_in_task_prompt: int,
        selected_category: str,
    ):
        samples = self.prompt_data["samples"]
        assert (
            selected_category in VALID_CATEGORY_NAMES
        ), f"Invalid category: {selected_category}. Valid categories: {VALID_CATEGORY_NAMES}"
        # Sample from the selected category
        selected_examples = random.sample(samples[selected_category], num_examples_in_task_prompt)
        random.shuffle(selected_examples)

        # Create the final prompt instruction with correct numbering
        task_instruction = self.prompt_data["task_instruction"] + "\n"

        sampled_examples = ""
        for i, example in enumerate(selected_examples, start=1):
            sampled_examples += f"{i}. {example}\n"

        formatted_examples = (
            '{number}. {{"{phrase_key}": "{phrase}", "{image_caption_key}": "{image_caption}", "{enhanced_phrase_key}":'
        )

        # add full stop to the end of the image_caption if it doesn't have one
        if not image_caption.endswith("."):
            image_caption += "."

        formatted_examples = formatted_examples.format(
            number=len(selected_examples) + 1,
            phrase_key="Phrase",
            phrase=phrase,
            image_caption_key="Full caption",
            image_caption=image_caption,
            enhanced_phrase_key="Enhanced phrase",
        )

        formatted_examples = f"{sampled_examples}{formatted_examples}"

        message = f"{task_instruction}{formatted_examples}"
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:  # if using Mixtral
            message = f"[INST] {message} [/INST]"

        return message

    def generate_by_phrase_list(self, phrase_list, caption_list, category="object"):
        all_prompts = [
            self._prepare_prompt(
                phrase=phrase,
                image_caption=caption,
                num_examples_in_task_prompt=5,
                selected_category=category,
            )
            for phrase, caption in zip(phrase_list, caption_list)
        ]

        outputs = self.generate_output_batch(all_prompts=all_prompts, max_length=32)
        outputs = [output[: output.find("}")] for output in outputs]
        outputs = [json.loads(output) for output in outputs]
        return outputs


class PhraseEnhancementProcessor:
    def __init__(self, llm: PromptEnhancerLLM):
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

    def process_image_edit_data(
        self, annotations: dict, llm_filtered_edits, batch_size: int = 8, save_every_n: int = 50
    ):
        logger.info(f"Total entries available for processing: {len(annotations)}")
        logger.info(f"Generating enhanced phrases and caching at {self.cache_file_path}")
        self.cached_phrase_data = self._load_cached_data(self.cache_file_path)
        copy_current_cache_file_as_backup_json(self.cache_file_path)

        tot_processed = 0
        image_ids_batch, uuids_batch, prompts_batch = [], [], []

        def process_batch(image_ids_batch, edit_ids_batch, prompts_batch):
            start_time = time.time()

            generated_outputs = self.llm.generate_output_batch(all_prompts=prompts_batch, num_return_sequences=1)
            logger.info(f"DEBUG: generated_outputs: {generated_outputs}")

            for image_id, edit_id, output in zip(image_ids_batch, edit_ids_batch, generated_outputs):
                try:
                    parsed_output = json.loads(output.rstrip("}"))
                except json.decoder.JSONDecodeError as e:
                    logger.info(f"Error parsing output: {output} with error: {e}")
                    continue

                self.cached_phrase_data[image_id][edit_id] = parsed_output

            end_time = time.time()
            total_time_taken = end_time - start_time
            logger.info(
                f"Total time taken to generate {len(generated_outputs)} phrases: {total_time_taken:.2f} seconds"
            )

        annotations = {
            k: v
            for k, v in annotations.items()
            if k not in self.cached_phrase_data
            or sum(len(values) for values in annotations[k].values()) != len(self.cached_phrase_data[k])
        }
        logger.info(f"Remaining entries to be processed: {len(annotations)}")

        for image_id in tqdm(annotations, desc="Generating enhanced phrases", position=0, leave=True):
            logger.info(
                f"len(annotations[image_id]): {sum(len(values) for values in annotations.get(image_id).values())} != len(self.cached_phrase_data[image_id]): {len(self.cached_phrase_data.get(image_id, []))}"
            )
            logger.debug(f"{annotations[image_id].values()} => {self.cached_phrase_data.get(image_id)}")
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            image_id = str(image_id)
            if image_id not in self.cached_phrase_data:
                self.cached_phrase_data[image_id] = {}

            curr_image_data = annotations[image_id]
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
                    and info["edit_id"] not in self.cached_phrase_data[image_id]
                ]
                logger.info(
                    f"Filtered {len(all_images_edit_ids) - len(curr_list_of_dicts)} edits out of {len(all_images_edit_ids)}"
                )
                for info in curr_list_of_dicts:
                    edit_id = info["edit_id"]
                    try:
                        edited_phrase, edited_caption, category, edit_id = (
                            info["edited_phrase"],
                            info.get("edited_region_phrase", info.get("edited_caption", "")),
                            info["category"],
                            info["edit_id"],
                        )
                        category = category.split("(")[0]
                    except Exception as e:
                        logger.info(f"Error parsing edit dict: {info} with error: {e}")
                        continue

                    if category not in VALID_CATEGORY_NAMES:
                        continue

                    prompt = self.llm._prepare_prompt(
                        phrase=edited_phrase,
                        image_caption=edited_caption,
                        num_examples_in_task_prompt=5,
                        selected_category=category,
                    )

                    image_ids_batch.append(image_id)
                    uuids_batch.append(edit_id)
                    prompts_batch.append(prompt)

                    if len(prompts_batch) == batch_size:
                        process_batch(image_ids_batch, uuids_batch, prompts_batch)
                        image_ids_batch, uuids_batch, prompts_batch = [], [], []
                        tot_processed += 1

            if (tot_processed + 1) % save_every_n == 0:
                save_to_json(self.cached_phrase_data, self.cache_file_path)

        # Process any remaining prompts
        if prompts_batch:
            process_batch(image_ids_batch, uuids_batch, prompts_batch)
            save_to_json(self.cached_phrase_data, self.cache_file_path)

        remove_current_cache_backup_file(self.cache_file_path)

    # FIXME: test carefully
    def process_image_layout_data(self, annotations, batch_size: int = 8, save_every_n: int = 50):
        logger.info(f"Total entries available for processing: {len(annotations)}")
        logger.info(f"Generating enhanced phrases and caching at {self.cache_file_path}")
        self.cached_phrase_data = self._load_cached_data(self.cache_file_path)

        tot_processed = 0
        image_ids_batch, phrases_batch, prompts_batch = [], [], []

        def process_batch(image_ids_batch, phrases_batch, prompts_batch):
            start_time = time.time()

            if any(isinstance(el, list) for el in prompts_batch):  # 2d list, flatten it
                prompts_batch = [item for sublist in prompts_batch for item in sublist]
            generated_outputs = self.llm.generate_output_batch(all_prompts=prompts_batch, num_return_sequences=1)
            # Chunk the generated outputs to match the original structure of phrases_batch
            iterator = iter(generated_outputs)
            generated_outputs_chunked = [list(islice(iterator, len(phrases))) for phrases in phrases_batch]

            logger.debug(f"DEBUG: generated_outputs: {generated_outputs_chunked}")

            for image_id, input_phrases, output_phrases in zip(
                image_ids_batch, phrases_batch, generated_outputs_chunked
            ):
                logger.info(f"INPUT PHRASES: {input_phrases}")
                curr_parse_data = {}
                for phrase, output in zip(input_phrases, output_phrases):
                    try:
                        parsed_output = json.loads(output.rstrip("}"))
                        curr_parse_data[phrase] = parsed_output
                    except json.decoder.JSONDecodeError as e:
                        logger.info(f"Error parsing output: {output} with error: {e}")
                        continue

                if len(curr_parse_data) == len(input_phrases):
                    self.cached_phrase_data[image_id] = curr_parse_data
                    logger.info(f"DEBUG: {image_id} => {curr_parse_data}")

            end_time = time.time()
            total_time_taken = end_time - start_time
            logger.info(
                f"Total time taken to generate {len(generated_outputs)} phrases: {total_time_taken:.2f} seconds"
            )

        for image_id in tqdm(annotations, desc="Generating enhanced phrases", position=0, leave=True):
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            curr_image_data = annotations[image_id]
            try:
                if isinstance(curr_image_data["bounding_boxes"], str):
                    bounding_box_data = eval(curr_image_data["bounding_boxes"])
                else:
                    bounding_box_data = curr_image_data["bounding_boxes"]
            except Exception as e:
                logger.info(f"Error parsing bounding box data: {curr_image_data['bounding_boxes']} with error: {e}")
                continue

            caption = curr_image_data["prompt"]
            object_names = [bbox[0] for bbox in bounding_box_data]
            object_names = list(set(object_names))

            curr_prompts = [
                self.llm._prepare_prompt(
                    phrase=phrase,
                    image_caption=caption,
                    num_examples_in_task_prompt=5,
                    selected_category="object",
                )
                for phrase in object_names
            ]

            image_ids_batch.append(image_id)
            phrases_batch.append(object_names)
            prompts_batch.append(curr_prompts)

            if len(image_ids_batch) == batch_size:
                process_batch(image_ids_batch, phrases_batch, prompts_batch)
                image_ids_batch, phrases_batch, prompts_batch = [], [], []
                tot_processed += 1

            if (tot_processed + 1) % save_every_n == 0:
                save_to_json(self.cached_phrase_data, self.cache_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "relation", "count"],
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
        default=None,
        help=f"Index of the chunk to process. Possible values: 0-{TOTAL_NUM_COCO_CHUNKS-1}.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="magic_prompt",
        help=f"Type of prompt to use. Possible values: instruct_pix2pix. Each choice represents a different strategy for generating prompts.",
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

    args = parser.parse_args()

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    prompt_type = args.prompt_type
    language_model_name = args.language_model_name  # "lmsys/vicuna-13b-v1.3" or "eachadea/vicuna-13b-1.1"
    dataset = args.dataset
    split = args.split
    chunk_index = args.chunk_index
    batch_size = args.batch_size
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()

    llm = PromptEnhancerLLM(
        language_model_name=language_model_name,
        prompt_type=prompt_type,
        device=device,
    )
    llm.load_pretrained_model_tokenizer()
    # load prompts that will be enhanced using llm
    output_dir = os.path.join(
        SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"llm_edits_{dataset}", "mistralaimixtral8x7binstructv0.1"
    )

    if dataset == "coco" and split == "train" and chunk_index is None:
        raise ValueError("Please provide a chunk index for the COCO train split.")

    src_prefix = "edit_suggestion" if dataset == "coco" else "llm_layout"
    if chunk_index is not None:
        inp_fpath = os.path.join(output_dir, f"{src_prefix}_{split}_chunked", f"chunk_{chunk_index}.json")
        output_fpath = os.path.join(output_dir, f"{prompt_type}_{split}_chunked", f"chunk_{chunk_index}.json")

    else:
        inp_fpath = os.path.join(output_dir, f"{src_prefix}_{split}.json")
        output_fpath = os.path.join(output_dir, f"{prompt_type}_{split}.json")

    with open(inp_fpath, "r") as f:
        annotations = json.load(f)

    enhance_proc = PhraseEnhancementProcessor(llm)
    enhance_proc.set_cache_file_path(output_fpath)

    if dataset == "coco":
        llm_filtered_edits = enhance_proc.load_cached_llm_filtered_edits_by_chunk_index(dataset, split, chunk_index)
        enhance_proc.process_image_edit_data(annotations, llm_filtered_edits, batch_size)
    elif dataset in ["relation", "counting"]:
        enhance_proc.process_image_layout_data(annotations, batch_size)

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from commons.constants import (LANGUAGE_MODEL_NAMES,
                               MISTRALAI_LANGUAGE_MODEL_NAMES,
                               SYNTH_DIFFUSE_DATA_DIR, TOTAL_NUM_COCO_CHUNKS)
from commons.logger import Logger
from evals.vlm.openai_client import OpenAIGPT

from ..utils.helpers import (group_outputs_for_batch_repeats, load_json_data,
                             save_to_json, set_random_seed)
from ..utils.llm_utils import BaseLLM

# Set up logger
logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()

VALID_CATEGORIES = ["object", "attribute"]


class EditInstructionFilterLLM(BaseLLM):
    """
    A class that uses a language model to generate an edit instruction and a new caption on an input caption.
    """

    @staticmethod
    def get_prompt_data_fpath(prompt_type):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root_dir, "promptsource", f"{prompt_type}.json")

    def load_embedding_model(self):
        self.embeddings_model = SentenceTransformer(
            "paraphrase-distilroberta-base-v1"
        )  # Load the pre-trained SBERT model

    def encode_prompt_demonstrations(self):
        samples = self.prompt_data["samples"]
        random.shuffle(samples)
        # get equal number of successful and unsuccessful examples
        self.sample_dict = defaultdict(lambda: defaultdict(list))
        for sample in samples:
            category = sample["Category"]
            reject = sample["IsReject"]
            if category not in VALID_CATEGORIES or reject not in ["YES", "NO"]:
                continue
            self.sample_dict[category][reject].append(sample)
        self.sample_embeddings = defaultdict(lambda: defaultdict(list))
        for category, category_samples in self.sample_dict.items():
            for reject, curr_samples in category_samples.items():
                curr_texts = [sample["EditedText"] for sample in curr_samples]
                self.sample_embeddings[category][reject] = self.embeddings_model.encode(
                    curr_texts, convert_to_tensor=True
                )

    def _prepare_prompt(
        self,
        input_text: Union[str, dict],
        num_examples_in_task_prompt: int,
    ):
        category = input_text.get("category")
        query_embedding = self.embeddings_model.encode(input_text["edited_caption"], convert_to_tensor=True)
        positive_scores = util.pytorch_cos_sim(query_embedding, self.sample_embeddings[category]["NO"])[0]
        sorted_indices = positive_scores.argsort(descending=True).tolist()
        successful_samples = [self.sample_dict[category]["NO"][i] for i in sorted_indices][
            : int(num_examples_in_task_prompt * 1.5)
        ]
        negative_scores = util.pytorch_cos_sim(query_embedding, self.sample_embeddings[category]["YES"])[0]
        sorted_indices = negative_scores.argsort(descending=True).tolist()
        unsuccessful_samples = [self.sample_dict[category]["YES"][i] for i in sorted_indices][
            : int(num_examples_in_task_prompt * 1.5)
        ]

        # randomize whether successful or unsuccessful samples are selected first or mixed; mixed being the least likely
        # Decide the operation: 0 or 1 selects and appends, 2 selects, appends, and then shuffles.
        operation = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        first_half = random.sample(
            successful_samples if operation != 1 else unsuccessful_samples, num_examples_in_task_prompt // 2
        )
        second_half = random.sample(
            unsuccessful_samples if operation != 1 else successful_samples, num_examples_in_task_prompt // 2
        )
        selected_examples = first_half + second_half
        if operation == 2:
            random.shuffle(selected_examples)

        # Create the final prompt instruction with correct numbering
        task_instruction = self.prompt_data[f"task_instruction_{category}"] + "\n"

        order = ["OriginalText", "EditedText", "FilterReason", "IsReject", "SuggestedCategory"]
        sampled_examples = ""
        for i, example in enumerate(selected_examples, start=1):
            example["IsReject"] = (
                example["IsReject"] + "." if not example["IsReject"].endswith(".") else example["IsReject"]
            )
            example_text = " ".join([f"{k}: {example[k]}" for k in order])
            sampled_examples += f"{i}. {example_text}\n"
        formatted_example = "{original_text_key}: {original_text} {edited_text_key}: {edited_text}"
        logger.debug(f"Sampled examples: {sampled_examples}")

        formatted_example = formatted_example.format(
            original_text_key="OriginalText",
            original_text=input_text.get("input_caption"),
            edited_text_key="EditedText",
            edited_text=input_text.get("edited_caption"),
        )
        test_invocation = "Now, do the following:\n"
        formatted_examples = f"{sampled_examples}{test_invocation}{formatted_example}"
        logger.debug(f"Formatted examples: {formatted_examples}")
        message = f"{task_instruction}{formatted_examples}"
        message = message.strip()
        if self.language_model_name in MISTRALAI_LANGUAGE_MODEL_NAMES:  # if using Mixtral
            message = f"[INST] {message} [/INST]"

        return message


class FilterProcessor:
    def __init__(self, llm: EditInstructionFilterLLM):
        self.llm = llm

    @staticmethod
    def _load_cached_data(json_file):
        if os.path.exists(json_file):
            logger.info(f"Loading existing cache from {json_file}")
            with open(json_file, "r") as f:
                return json.load(f)

        return defaultdict(dict)

    def set_cache_file_path(self, cache_file_path: str):
        if not os.path.exists(os.path.dirname(cache_file_path)):
            logger.warning(f"Cache directory does not exist. Creating directory: {os.path.dirname(cache_file_path)}")
            os.makedirs(os.path.dirname(cache_file_path))
        self.cache_file_path = cache_file_path

    def self_consistency_chain_of_thought(self, generated_outputs: List[str]):
        logger.debug(f"Generated outputs: {generated_outputs}")
        # Extract the votes and their indices from the generated outputs
        votes = []
        for i, output in enumerate(generated_outputs):
            if "Reject: YES" in output:
                votes.append((i, "YES"))
            elif "Reject: NO" in output:
                votes.append((i, "NO"))
            else:
                votes.append((i, "YES"))

        # Count the votes and determine the majority
        vote_counts = Counter(vote for _, vote in votes)
        majority_vote = max(vote_counts, key=vote_counts.get)
        logger.debug(f"Votes: {votes} => Majority vote: {majority_vote}")

        # Get the indices of the majority votes
        majority_indices = [i for i, vote in votes if vote == majority_vote]

        # Return a random output from the majority vote indices
        return generated_outputs[random.choice(majority_indices)]

    def process_image_edit_data(self, annotations: dict, batch_size: int = 8, save_every_n: int = 1):
        self.cached_filtered_data = self._load_cached_data(self.cache_file_path)
        image_ids_batch, uuids_batch, info_batch = [], [], []
        num_return_sequences = 3
        openai = OpenAIGPT()

        def process_batch_openai(image_ids_batch, edit_ids_batch, info_batch):
            start_time = time.time()
            for image_id, edit_id, info in zip(image_ids_batch, edit_ids_batch, info_batch):
                prompt = self.llm._prepare_prompt(
                    input_text=info,
                    num_examples_in_task_prompt=12,
                )
                raw_response = openai.get_response(prompt, temperature=0.2, is_json=False)
                output = raw_response.choices[0].message.content
                print(output)
                logger.debug(f"{image_id}:{info.get('edited_caption')} => {output}")
                if image_id not in self.cached_filtered_data:
                    self.cached_filtered_data[image_id] = {}

                self.cached_filtered_data[image_id][edit_id] = {
                    "reject": "NO" if "Reject: NO" in output else "YES",
                    "output": output.strip(),
                }
                print(self.cached_filtered_data[image_id][edit_id])

            end_time = time.time()
            total_time_taken = end_time - start_time
            logger.info(
                f"Total time taken to generate outputs (for batch size {batch_size}) : {total_time_taken:.2f} seconds"
            )

        def process_batch(image_ids_batch, edit_ids_batch, info_batch):
            start_time = time.time()
            # !!! IMPORTANT HACK !!!
            # WARNING: we're repeating the prompt for each example to increase diversity in the task prompt
            REPEAT_PROMPT_TIMES = 3
            all_generated_outputs = []

            for _ in range(REPEAT_PROMPT_TIMES):
                prompts_batch = []
                for info in info_batch:
                    logger.debug(f"Input text: {info}")
                    prompt = self.llm._prepare_prompt(
                        input_text=info,
                        num_examples_in_task_prompt=8,
                    )
                    logger.debug(f"Prompt: {prompt}")
                    prompts_batch.append(prompt)

                # Generate unique r_temp and r_top_k for this batch
                r_temp = random.uniform(0.75, 0.9)
                r_top_k = random.randint(40, 70)

                generated_outputs = self.llm.generate_output_batch(
                    all_prompts=prompts_batch,
                    num_return_sequences=num_return_sequences,
                    max_length=156,
                    temperature=r_temp,
                    top_k=r_top_k,
                    stop=["OriginalText:"],  # stop at the end of the original text
                )
                all_generated_outputs.extend(generated_outputs)

            grouped_outputs = group_outputs_for_batch_repeats(
                all_generated_outputs, batch_size, REPEAT_PROMPT_TIMES, num_return_sequences
            )
            final_outputs = [self.self_consistency_chain_of_thought(chunk_list) for chunk_list in grouped_outputs]
            for image_id, edit_id, info, output in zip(image_ids_batch, edit_ids_batch, info_batch, final_outputs):
                if image_id not in self.cached_filtered_data:
                    self.cached_filtered_data[image_id] = {}
                self.cached_filtered_data[image_id][edit_id] = {
                    "reject": "YES" if "Reject: YES" in output else "NO",
                    "output": output.strip(),
                }
                logger.info(
                    f"{image_id}:{info.get('edited_caption')} => {self.cached_filtered_data[image_id][edit_id]}"
                )

            end_time = time.time()
            total_time_taken = end_time - start_time
            logger.info(
                f"Total time taken to generate {len(all_generated_outputs)} outputs (for batch size {batch_size}) : {total_time_taken:.2f} seconds"
            )

        annotation_changed = False
        for i, image_id in enumerate(tqdm(annotations, desc="Filtering edit instructions")):
            if not self.llm:
                raise Exception("LLM is not set. Please set LLM before generating captions.")

            image_id = str(image_id)
            curr_image_data = annotations[image_id]
            for input_caption, curr_list_of_dicts in curr_image_data.items():
                for info in curr_list_of_dicts:
                    edit_id = info["edit_id"]
                    if edit_id in self.cached_filtered_data.get(image_id, {}):
                        logger.info(f"Skipping {image_id}:{edit_id} as it already exists in the cache")
                        continue

                    try:
                        category = info["category"]
                        category = category.split("(")[0]
                        info["category"] = category
                    except Exception as e:
                        logger.info(f"Error parsing edit dict: {info} with error: {e}")
                        continue

                    if category not in VALID_CATEGORIES:  # must be either object or attribute
                        logger.debug(f"Skipping {image_id}:{edit_id} as it has invalid category: {category}")
                        continue

                    info.pop("edit_id")
                    image_ids_batch.append(image_id)
                    uuids_batch.append(edit_id)
                    info_batch.append({"input_caption": input_caption, **info})

                    if len(info_batch) == batch_size:
                        process_batch(image_ids_batch, uuids_batch, info_batch)
                        # process_batch_openai(image_ids_batch, uuids_batch, info_batch)
                        image_ids_batch, uuids_batch, info_batch = [], [], []
                        annotation_changed = True

            if (i + 1) % save_every_n == 0 and annotation_changed:
                save_to_json(self.cached_filtered_data, cache_file_path)
                annotation_changed = False

        # Process any remaining prompts
        if info_batch:
            process_batch(image_ids_batch, uuids_batch, info_batch)
            # process_batch_openai(image_ids_batch, uuids_batch, info_batch)
            save_to_json(self.cached_filtered_data, cache_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "vsr"],
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
        required=False,
        help=f"Index of the chunk to process. Possible values: 0-{TOTAL_NUM_COCO_CHUNKS-1}.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="auto_filter_edit_instruct",
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

    prompt_type = args.prompt_type
    language_model_name = args.language_model_name  # "lmsys/vicuna-13b-v1.3" or "eachadea/vicuna-13b-1.1"
    dataset = args.dataset
    split = args.split
    chunk_index = args.chunk_index
    batch_size = args.batch_size
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()

    llm = EditInstructionFilterLLM(
        language_model_name=language_model_name,
        prompt_type=prompt_type,
        device=device,
    )
    llm.load_embedding_model()
    llm.encode_prompt_demonstrations()
    llm.load_pretrained_model_tokenizer()
    # load prompts that will be enhanced using llm
    dir_path = os.path.join(
        SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"llm_edits_{dataset}", "mistralaimixtral8x7binstructv0.1"
    )

    if dataset == "coco" and split == "train" and chunk_index is None:
        raise ValueError("Chunk index is required for COCO train split.")

    if chunk_index:
        fname = os.path.join(dir_path, f"edit_suggestion_{split}_chunked", f"chunk_{chunk_index}.json")
        cache_file_path = os.path.join(
            dir_path, f"edit_instruction_filtered_{split}_chunked", f"chunk_{chunk_index}.json"
        )
    else:
        fname = os.path.join(dir_path, f"edit_suggestion_{split}.json")
        cache_file_path = os.path.join(dir_path, f"edit_instruction_filtered_{split}.json")

    with open(fname, "r") as f:
        annotations = json.load(f)

    enhance_proc = FilterProcessor(llm)
    enhance_proc.set_cache_file_path(cache_file_path)
    enhance_proc.process_image_edit_data(annotations, batch_size)

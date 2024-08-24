import argparse
import os
import random
import re

import torch

from commons.constants import (LANGUAGE_MODEL_NAMES, SYNTH_DIFFUSE_DATA_DIR,
                               SYNTH_ONLY_CATEGORIES, TOTAL_NUM_COCO_CHUNKS)
from commons.logger import Logger

from ..llm_agent.edit_instructgen_llm import (EditInstructgenBootstrap,
                                              EditInstructgenFromCaption)
from ..utils.helpers import load_coco_captions, load_vsr_captions
from ..utils.llm_utils import (DiffEditLLM, EditsGenLLM, GroundedLLM,
                               InstructPix2PixLLM)

logger = Logger.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)


def create_llm_agent(model_key, language_model_name, prompt_type, device):
    model_map = {
        "llm_layout": GroundedLLM,
        "edit_suggestion": EditsGenLLM,
        "hive": InstructPix2PixLLM,
        "instruct_pix2pix": InstructPix2PixLLM,
        "diffedit": DiffEditLLM,
        "sdedit": InstructPix2PixLLM,
    }

    ModelClass = model_map.get(model_key)

    # Special case for hive and sdedit
    if model_key in ["hive", "sdedit"]:
        return ModelClass(language_model_name=language_model_name, prompt_type="instruct_pix2pix", device=device)

    return ModelClass(language_model_name=language_model_name, prompt_type=prompt_type, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "vsr", "relation", "counting"],
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
        help=f"Index of the COCO chunk to process. Possible values: 0-{TOTAL_NUM_COCO_CHUNKS-1}.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="edit_instructgen_from_caption",
        help=f"Type of prompt to use. Possible values: instruct_pix2pix. Each choice represents a different strategy for generating prompts.",
    )
    parser.add_argument(
        "--language_model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
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

    args.prompt_type = "llm_layout" if args.dataset in SYNTH_ONLY_CATEGORIES else args.prompt_type
    dataset_name = args.dataset
    split = args.split
    chunk_index = args.chunk_index
    prompt_type = args.prompt_type
    batch_size = args.batch_size
    language_model_name = args.language_model_name  # "lmsys/vicuna-13b-v1.3" or "eachadea/vicuna-13b-1.1"

    # load coco annotations from filepath `captions_train_2014.json`
    if dataset_name == "coco":
        coco_annotations = load_coco_captions(split=split, chunk_index=chunk_index)
        random.shuffle(coco_annotations)

    elif args.dataset == "vsr":
        coco_annotations = load_vsr_captions(split=split)
        random.shuffle(coco_annotations)

    max_gen_try = 3
    num_samples_per_image = 10 if dataset_name == "vsr" else 5
    language_model_name_str = re.sub(r"[/\-]", "", language_model_name).lower()
    if language_model_name in ["Mixtral-8x7B-Instruct-v0.1", "teknium/OpenHermes-2.5-Mistral-7B"]:
        language_model_name_str = "mistralaimixtral8x7binstructv0.1"

    if dataset_name == "coco" and split == "train" and chunk_index is None:
        raise ValueError("Chunk index is required for COCO train split.")

    if chunk_index is not None:
        cache_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset_name}",
            language_model_name_str,
            f"{prompt_type}_{split}_chunked",
            f"chunk_{chunk_index}.json",
        )
    else:
        cache_fpath = os.path.join(
            SYNTH_DIFFUSE_DATA_DIR,
            "prompt_resources",
            f"llm_edits_{dataset_name}",
            language_model_name_str,
            f"{prompt_type}_{split}.json",
        )

    llm = create_llm_agent(prompt_type, language_model_name, prompt_type, device)

    if dataset_name in SYNTH_ONLY_CATEGORIES:
        max_num_samples = 100000 if split == "train" else 50000
        edit_gen_manager = EditInstructgenBootstrap(cache_fpath, max_num_samples=max_num_samples)
        if edit_gen_manager.is_cache_complete():
            exit(0)

        logger.info(
            "Incomplete cache found. Initiating full dataset caching for llm generation. This might take some time..."
        )
        llm.load_pretrained_model_tokenizer()
        llm.load_valid_coco_nouns()
        edit_gen_manager.set_llm(llm)
        edit_gen_manager.cache_full_dataset_bootstrap_edit_instructions(
            dataset_name, num_return_sequences=2, batch_generate=True
        )

    else:
        edit_gen_manager = EditInstructgenFromCaption(cache_fpath)
        if edit_gen_manager.is_cache_complete(coco_annotations, num_samples_per_image=num_samples_per_image):
            logger.info("Cache is complete. Exiting...")
            exit(0)

        logger.info(
            "Incomplete cache found. Initiating full dataset caching for llm generation. This might take some time..."
        )
        llm.load_pretrained_model_tokenizer()
        edit_gen_manager.set_llm(llm)
        edit_gen_manager.cache_full_dataset_coco_edit_instructions_batch(
            coco_annotations,
            dataset=dataset_name,
            max_gen_try=max_gen_try,
            num_samples_per_image=num_samples_per_image,
            batch_size=batch_size,
            num_return_sequences=2,
            num_examples_in_task_prompt=16,
        )

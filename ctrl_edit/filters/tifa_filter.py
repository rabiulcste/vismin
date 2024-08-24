import argparse
import json
import os
import random
import time

import numpy as np
import torch
from groundingdino.util.inference import load_image, predict
from PIL import Image, ImageDraw
from torchvision.ops import box_convert
from tqdm import tqdm

from tifa.tifascore import UnifiedQAModel

from ..utils.constants import (SYNTH_DIFFUSE_DATA_DIR, TOTAL_NUM_COCO_CHUNKS,
                               VIM_DATA_DIR)
from ..utils.helpers import (FileLocker, crop_image_region_from_bbox_list,
                             get_coco_path_by_image_id, load_json_data,
                             remove_duplicate_dict_entries_by_key,
                             save_to_json, set_random_seed)
from ..utils.logger import Logger
from .vqa_models import VQAModelForTifa

logger = Logger.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed()


class GroundingGenerator:
    def __init__(
        self,
        repo_id,
        ckpt_filename,
        ckpt_config_filename,
        device="cuda",
    ):
        logger.info(
            f"Initializing {self.__class__.__name__} with repo_id: {repo_id} and ckpt_filename: {ckpt_filename}"
        )
        self.device = device
        self.dino_model = self.load_model_hf(repo_id, ckpt_filename, ckpt_config_filename)

    def load_model_hf(self, repo_id, filename, ckpt_config_filename):
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from huggingface_hub import hf_hub_download

        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = self.device
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location="cpu")
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        logger.info(f"Model loaded from {cache_file} \n => {log}")
        return model

    def generate_grounding(self, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu",
        )

        logger.info(f"no of boxes: {len(boxes)}, phrases: {phrases}, boxes: {boxes}")

        # check or remove boxes that capture the entire image
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

        # Keep only the boxes that cover less than 85% of the image
        boxes_area_tensor = boxes_area
        mask = boxes_area_tensor < 0.85
        indices = np.where(mask)[0].tolist()
        boxes_xyxy = boxes_xyxy[mask]  # making sure boxes is a tensor
        phrases = [phrases[i] for i in indices]

        logger.info(f"no of boxes (post-filtering): {len(boxes_xyxy)}, phrases: {phrases}, boxes: {boxes_xyxy}")
        return boxes_xyxy, phrases


def draw_rectangle_image_region(image_fpath: str, bboxes: list) -> Image.Image:
    image = Image.open(image_fpath).convert("RGB")

    if not bboxes:
        return image

    w, h = image.size
    bboxes = bboxes * np.array([w, h, w, h])

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
        # draw a rectangle on the image to highlight the region and return the image
        draw = ImageDraw.Draw(image)
        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)
    except Exception as e:
        logger.info(f"Error cropping image: {e}")

    return image


class ImageVQAVerifier:
    def __init__(self, tifa_scorer: VQAModelForTifa, unifiedqa_model: UnifiedQAModel):
        self.tifa_scorer = tifa_scorer
        self.unifiedqa_model = unifiedqa_model
        self.file_locker = FileLocker()

    def get_prompt_data_fpath(self, prompt_type):
        return os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"{prompt_type}.json")

    def score_and_evaluate_generated_qa(
        self, dataset: str, chunk_index: int, split: str, add_shuffled_choices: bool = False
    ):
        data_dir_root = os.path.join(VIM_DATA_DIR, f"{dataset}_sdxl_edited_{split}")
        # this grounding generator is temporary hack as some data is missing bounding boxes annotations
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swint_ogc.pth"
        ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"

        grounding_generator = GroundingGenerator(
            repo_id=ckpt_repo_id,
            ckpt_filename=ckpt_filename,
            ckpt_config_filename=ckpt_config_filename,
        )
        if chunk_index is not None:
            annotations_fpath = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR,
                "prompt_resources",
                f"qa_annotations_{dataset}_{split}_chunked",
                f"chunk_{chunk_index}.json",
            )
        else:
            annotations_fpath = os.path.join(
                SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"qa_annotations_{dataset}_{split}.json"
            )
        qa_annotations_dict = load_json_data(annotations_fpath)
        if qa_annotations_dict is None:
            logger.error(
                f"Failed to load qa_annotations_dict from {annotations_fpath}. "
                "This file is required for the filtering process. "
                "Please ensure the file exists and is accessible. "
                "If not, you may need to first create question-answer pairs before running this filtering process."
            )
            return

        image_directories = os.listdir(data_dir_root)
        logger.info(f"Directory: {data_dir_root}")
        logger.info(f"Number of image directories available for processing: {len(image_directories)}")
        # validate image directories with qa_annotations_dict
        image_directories = [image_id for image_id in image_directories if str(image_id) in qa_annotations_dict]
        logger.info(f"Number of image directories with question-answer pairs annotations: {len(image_directories)}")
        random.shuffle(image_directories)

        tot_processed, success_count, already_in_cache, less_than_threshold = 0, 0, 0, 0
        for image_id in tqdm(image_directories, desc="Processing image directories"):
            curr_image_dir = os.path.join(data_dir_root, image_id)
            output_fpath = os.path.join(curr_image_dir, "annotations.json")
            image_data_dict = load_json_data(output_fpath)
            if image_data_dict is None:
                continue

            annotations = image_data_dict["annotations"]
            annotations_changed = False  # Initialize change flag

            # check if the edit_id is present in qa_annotations_dict
            if not any(qa_annotations_dict.get(str(image_id), {}).get(info["edit_id"]) for info in annotations):
                logger.info(f'Skipping image_id {image_id} as no edit_id found in "qa_annotations_coco_{split}.json"')
                continue

            with self.file_locker.locked(curr_image_dir) as lock_acquired:
                if not lock_acquired:
                    logger.warning(f"Skipping image id: {image_id} as another process is working on it.")
                    continue

                logger.debug(f"Processing image_id {image_id}")
                start_time = time.time()

                for info in annotations:
                    edit_id = info["edit_id"]
                    # categoy = info["category"]
                    # if "attribute" not in categoy:
                    #     continue

                    # Skip if tifa_score and region_tifa_score is already computed for all edits
                    if all("region_tifa_score" in score_dict for score_dict in info["scores"].values()):
                        logger.info(f"Skipping scoring for image_id:edit_id = {image_id}:{edit_id} as already scored")
                        already_in_cache += len(info["scores"])
                        continue

                    qa_for_edit = qa_annotations_dict.get(str(image_id), {}).get(edit_id, {})
                    question_answer_pairs_fullcap = qa_for_edit.get("generated_questions_full_caption", [])
                    question_answer_pairs_regcap = qa_for_edit.get("generated_questions_region_caption", [])
                    # question_answer_pairs_fullcap += question_answer_pairs_regcap
                    question_answer_pairs_fullcap = remove_duplicate_dict_entries_by_key(
                        question_answer_pairs_fullcap, "question"
                    )

                    if add_shuffled_choices:
                        qap_fullcap_shuffled = [
                            {
                                "question": qap["question"],
                                "choices": qap["choices"][::-1]
                                if len(qap["choices"]) == 2
                                else self.shuffle_choices(qap["choices"]),
                                "answer": qap["answer"],
                            }
                            for qap in question_answer_pairs_fullcap
                        ]

                        qap_regcap_shuffled = [
                            {
                                "question": qap["question"],
                                "choices": qap["choices"][::-1]
                                if len(qap["choices"]) == 2
                                else self.shuffle_choices(qap["choices"]),
                                "answer": qap["answer"],
                            }
                            for qap in question_answer_pairs_regcap
                        ]
                        question_answer_pairs_fullcap += qap_fullcap_shuffled
                        question_answer_pairs_regcap += qap_regcap_shuffled

                    if not question_answer_pairs_fullcap or not question_answer_pairs_regcap:
                        logger.info(f"No question-answer pairs found for edit_id: {info['edit_id']}")
                        continue

                    bounding_boxes = info.get("bounding_boxes", [])
                    if not bounding_boxes:
                        # a function to add bounding boxes to the image_data_dict
                        coco_split = "val" if split == "validation" else split
                        image_path = get_coco_path_by_image_id(split=coco_split, image_id=image_id)
                        input_image, transformed_image = load_image(image_path)
                        bounding_boxes, phrases = grounding_generator.generate_grounding(
                            image=transformed_image,
                            text_prompt=info["input_phrase"],
                        )
                        bounding_boxes = bounding_boxes.tolist()
                        info["bounding_boxes"] = bounding_boxes
                        annotations_changed = True

                    # Proceed based on the selected scoring mode
                    for image_path, score_dict in info["scores"].items():
                        if all(key in score_dict for key in ("tifa_score", "region_tifa_score")):
                            # Skip if the required score is already computed
                            logger.info(
                                f"Skipping scoring for image_id:edit_id = {image_id}:{edit_id} as already scored"
                            )
                            already_in_cache += 1
                            continue

                        # FIXME: this should be "region_tifa_score" instead of "tifa_score"
                        if "tifa_score" in score_dict and score_dict["tifa_score"] < 0.5:
                            logger.info(
                                f"Since old tifa_score is < 0.5, skipping scoring for image_id:edit_id = {image_id}:{edit_id}"
                            )
                            less_than_threshold += 1
                            continue

                        logger.info(f"Scoring image_path: {image_path}")
                        logger.info(f"Image caption: {info['edited_caption']}")
                        logger.info(f"Bounding boxes: {bounding_boxes}")
                        image = Image.open(image_path).convert("RGB")
                        cropped_image = crop_image_region_from_bbox_list(image_path, bounding_boxes)

                        try:
                            result_dict = self.tifa_scorer.get_tifa_score(question_answer_pairs_fullcap, image)
                            region_result_dict = self.tifa_scorer.get_tifa_score(
                                question_answer_pairs_regcap, cropped_image
                            )
                        except Exception as e:
                            logger.error(f"Error scoring edit_id: {edit_id} with error {e}")
                            continue

                        region_result_dict["region_tifa_score"] = region_result_dict.pop("tifa_score")
                        result_dict["choices_shuffled_bool"] = add_shuffled_choices
                        result_dict["region_prefix_bool"] = True
                        result_dict["vqa_model"] = self.tifa_scorer.model_name_or_path
                        logger.info(
                            f"Result for tifa_score: {result_dict.get('tifa_score')} and region_tifa_score: {region_result_dict.get('region_tifa_score')}"
                        )
                        info["scores"][image_path].update(result_dict)
                        info["scores"][image_path].update(region_result_dict)
                        success_count += result_dict.get("tifa_score", 0) == 1
                        tot_processed += 1
                        annotations_changed = True

            if annotations_changed:
                save_to_json(image_data_dict, output_fpath)

                end_time = time.time()
                total_time_taken = end_time - start_time
                logger.info(
                    f"Total time taken to verify {len(image_data_dict['annotations'])} images: {total_time_taken:.2f} seconds"
                )
                logger.info(f"Perc. success: {success_count / tot_processed * 100:.2f}%")
                logger.info(f"Already in cache: {already_in_cache} | Less than threshold: {less_than_threshold}")

    @staticmethod
    def shuffle_choices(choices):
        shuffled = choices[:]
        random.shuffle(shuffled)
        return shuffled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question generation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
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
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    args = parser.parse_args()
    dataset = args.dataset
    split = args.split
    chunk_index = args.chunk_index

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    model_name_or_path = "HuggingFaceM4/idefics2-8b"  # or liuhaotian/llava-v1.6-34b or liuhaotian/llava-v1.5-13b

    # if split == "train":
    #     model_name_or_path = (
    #         "llava-hf/llava-v1.6-mistral-7b-hf"  # or liuhaotian/llava-v1.6-34b or liuhaotian/llava-v1.5-13b
    #     )
    # else:
    #     model_name_or_path = "liuhaotian/llava-v1.6-34b"

    tifa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=8)
    unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    image_verify_engine = ImageVQAVerifier(tifa_scorer, unifiedqa_model)
    image_verify_engine.score_and_evaluate_generated_qa(dataset, chunk_index, split)

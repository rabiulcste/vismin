import argparse
import json
import os
import random

import PIL
import torch
from PIL import Image
from tqdm import tqdm

from .filters.vqa_models import VQAModelForTifa
from .utils.constants import SYNTH_DIFFUSE_DATA_DIR, VIM_DATA_DIR
from .utils.helpers import load_json_data, set_random_seed
from .utils.logger import Logger
from .utils.spatial_relation_utils import get_xyxy_from_xywh

logger = Logger.get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()

from tifa.tifascore import UnifiedQAModel, filter_question_and_answers


class CountingImageVerify:
    def __init__(self, split: str, vqa_scorer: VQAModelForTifa):
        self.split = split
        self.vqa_scorer = vqa_scorer
        self.unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-3b-1363200")

    def load_qa_annotations(self, split: str):
        qa_annotations = {}
        qa_fpath = os.path.join(SYNTH_DIFFUSE_DATA_DIR, "prompt_resources", f"qa_annotations_counting_{split}.json")
        with open(qa_fpath, "r") as f:
            qa_annotations = json.load(f)

        return qa_annotations

    def run(self, image_data_dir: str):
        # Load QA annotations
        qa_annotations = self.load_qa_annotations(self.split)

        image_directories = os.listdir(image_data_dir)
        random.shuffle(image_directories)
        logger.info(f"Total number of images found: {len(image_directories)}")

        success_count, tot_processed = 0, 0
        for image_id in tqdm(image_directories, desc="Counting image verify"):
            annotation_fpath = os.path.join(image_data_dir, image_id, "annotations.json")
            data = load_json_data(annotation_fpath)
            if data is None:
                continue

            annotations = data["annotations"]
            prompt = annotations["prompt"]
            inp_image_path = annotations["generated_images"][0]
            inp_image = Image.open(inp_image_path)
            bounding_boxes = annotations["bounding_boxes"]

            generated_questions = qa_annotations.get(image_id, {}).get("generated_questions")
            if generated_questions is None:
                continue

            edit_instructions = annotations.get("edit_instructions")
            if edit_instructions is None:
                continue

            annotation_changed = False
            for info in edit_instructions:
                """
                {'remove_object_names': ['taxi'], 'remove_object_bbox_indices': [[50, 150, 150, 100]], 'input_phrase': 'four taxis', 'edited_phrase':
                'three taxis', 'edited_caption': 'A scene with three taxis', 'edit_id': 'e66da9'}
                """
                edit_id = info["edit_id"]
                edited_image_path = os.path.join(image_data_dir, image_id, f"{edit_id}.png")
                try:
                    edited_image = Image.open(edited_image_path).convert("RGB")
                except Exception as e:
                    continue
                old_tifa_score = annotations["scores"].get(edited_image_path, {}).get("tifa_score")
                if old_tifa_score < 0.9:
                    continue
                llm_qa_verify_bool = annotations["scores"].get(edited_image_path, {}).get("llm_qa_verify", False)
                if llm_qa_verify_bool:
                    logger.info(f"Skipping {edited_image_path} as it has already been verified")
                    continue

                curr_bounding_boxes = [
                    bbox for bbox in bounding_boxes if bbox[1] not in info["remove_object_bbox_indices"]
                ]
                assert len(curr_bounding_boxes) == len(bounding_boxes) - len(
                    info["remove_object_bbox_indices"]
                ), f"mismatch in bounding boxes ({len(curr_bounding_boxes)} vs {len(bounding_boxes) - len(info['remove_object_bbox_indices'])})"
                if len(curr_bounding_boxes) < 3:
                    continue
                logger.info(f"PROMPT: {info['edited_caption']}")
                logger.info(f"IMAGE PROMPT: {edited_image_path}")

                tifa_result_dict_final = {"tifa_score": [], "question_details": []}
                for obj_name, obj_bbox in curr_bounding_boxes:
                    keys = [obj_name, f"a {obj_name}", f"an {obj_name}"]
                    question_answer_pairs = next(
                        (generated_questions.get(key) for key in keys if generated_questions.get(key) is not None), None
                    )
                    # "clearly visible" if in the question we don't want those questions
                    if question_answer_pairs is None:
                        continue
                    excluded_phrases = ["clearly visible", "deformed", "look clear"]
                    question_answer_pairs = [
                        qa_pair
                        for qa_pair in question_answer_pairs
                        if not any(phrase in qa_pair["question"] for phrase in excluded_phrases)
                    ]

                    if obj_name.startswith(("a ", "an ")):
                        caption = f"A photo of {obj_name}."
                    else:
                        caption = f"A photo of a {obj_name}."
                    question_answer_pairs = filter_question_and_answers(
                        self.unifiedqa_model, question_answer_pairs, caption
                    )
                    if len(question_answer_pairs) == 0:
                        continue
                    # increase bbox size by 10%, it's in xywh format
                    obj_bbox = [
                        obj_bbox[0] - 0.05 * obj_bbox[2],
                        obj_bbox[1] - 0.05 * obj_bbox[3],
                        obj_bbox[2] * 1.1,
                        obj_bbox[3] * 1.1,
                    ]

                    object_crop = edited_image.crop(get_xyxy_from_xywh(obj_bbox))
                    # resize the image to 224x224 if its width or height is smaller than 224
                    if object_crop.size[0] < 224 or object_crop.size[1] < 224:
                        object_crop = object_crop.resize((224, 224), PIL.Image.ANTIALIAS)

                    # object_crop.save(f"diffused_generator/output/counting_debug/{obj_name}.png")
                    # print(f"{obj_name} => {obj_bbox} # {len(qa_pairs)}")
                    tifa_result_dict = vqa_scorer.get_tifa_score(question_answer_pairs, object_crop)
                    tifa_result_dict_final["tifa_score"].append(tifa_result_dict["tifa_score"])
                    tifa_result_dict_final["question_details"].append(tifa_result_dict["question_details"])

                if len(tifa_result_dict_final["tifa_score"]) == 0:
                    continue

                logger.info(f"tifa results (all): {tifa_result_dict_final['tifa_score']}")
                tifa_result_dict_final["tifa_score"] = sum(tifa_result_dict_final["tifa_score"]) / len(
                    tifa_result_dict_final["tifa_score"]
                )
                q_details_failed = []
                for q_details in tifa_result_dict_final["question_details"]:
                    for key, val in q_details.items():
                        if val["scores"] != 1:
                            q_details_failed.append(val)
                # if len(q_details_failed) == 0 and tifa_result_dict_final["tifa_score"] < 0.9:
                #     logger.warning(json.dumps(tifa_result_dict_final["question_details"], indent=2))

                print(f"Failed questions: {q_details_failed}")
                logger.info(f"tifa results: {tifa_result_dict_final['tifa_score']}")
                success_count += tifa_result_dict_final["tifa_score"] > 0.9
                tot_processed += 1
                annotations["scores"][edited_image_path] = tifa_result_dict_final
                annotations["scores"][edited_image_path]["llm_qa_verify"] = True
                annotation_changed = True

            if tot_processed > 0 and tot_processed % 10 == 0:
                logger.info(f"Perc. success: {success_count / tot_processed * 100:.2f}%")

            if annotation_changed:
                with open(annotation_fpath, "w") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Saved annotations to {annotation_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use. Possible values: train, validation, test.",
    )
    args = parser.parse_args()
    split = args.split

    args_dict_formatted = ", ".join([f"`{attr}`: {getattr(args, attr)}" for attr in vars(args)])
    logger.info(f"Arguments: ({args_dict_formatted})")

    layout_image_data_dir = os.path.join(VIM_DATA_DIR, f"bootstrap_layout_counting_{split}")
    model_name_or_path = "HuggingFaceM4/idefics2-8b"  # or liuhaotian/llava-v1.6-34b or vikhyatk/moondream2 or llava-hf/llava-v1.6-mistral-7b-hf or HuggingFaceM4/idefics2-8b
    vqa_scorer = VQAModelForTifa(model_name_or_path, load_in_Nbit=None)

    count_verify = CountingImageVerify(split=split, vqa_scorer=vqa_scorer)
    count_verify.run(layout_image_data_dir)
